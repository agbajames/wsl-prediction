"""
Offline Dixon-Coles/champion-family configuration variants.

The variants in this module are intentionally small and predeclared. They wrap
the existing champion evaluation adapter without changing production prediction
behaviour or the frozen ``champion_dc_xg`` reference model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from evaluation.backtesting import BacktestConfig, config_to_dict, run_backtest_for_model
from model.wsl_xg_model import ModelConfig
from models.champion_dc_xg import ChampionDCXGModel
from scripts.run_evaluation_report import build_report_summary, render_markdown_report
from scripts.run_model_comparison import attach_actual_results


@dataclass(frozen=True)
class DixonColesVariantSpec:
    """Serializable specification for one offline champion-family variant."""

    model_name: str
    description: str
    config_overrides: dict[str, Any]
    rho_behavior: str = "fixed_default"

    @property
    def model_family(self) -> str:
        return "xg_dixon_coles_config_variant"

    @property
    def model_version(self) -> str:
        return "v1"


@dataclass
class DixonColesVariantModel:
    """Evaluation-compatible wrapper around ``ChampionDCXGModel``."""

    spec_def: DixonColesVariantSpec

    _champion: ChampionDCXGModel | None = None

    @property
    def name(self) -> str:
        return self.spec_def.model_name

    @property
    def family(self) -> str:
        if self.name == "champion_dc_xg":
            return "xg_dixon_coles_poisson"
        return self.spec_def.model_family

    @property
    def version(self) -> str:
        return self.spec_def.model_version

    @property
    def spec(self) -> dict[str, Any]:
        return self.export_config()

    def fit(self, played: pd.DataFrame) -> DixonColesVariantModel:
        self._champion = ChampionDCXGModel(
            config=build_model_config(self.spec_def),
            fit_rho_when_config_none=self.spec_def.rho_behavior in {"fit_when_none", "fit"},
        ).fit(played)
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        if self._champion is None:
            raise RuntimeError("DixonColesVariantModel must be fitted before predict().")
        predictions = self._champion.predict(fixtures).copy()
        _normalise_probability_columns(predictions)
        predictions["model_name"] = self.name
        predictions["model_family"] = self.family
        predictions["model_version"] = self.version
        predictions["variant_description"] = self.spec_def.description
        return predictions

    def export_config(self) -> dict[str, Any]:
        config = asdict(build_model_config(self.spec_def))
        config["rho_grid"] = list(config["rho_grid"])
        return {
            "model_name": self.name,
            "model_family": self.family,
            "model_version": self.version,
            "description": self.spec_def.description,
            "config": config,
            "config_overrides": dict(self.spec_def.config_overrides),
            "rho_behavior": self.spec_def.rho_behavior,
            "offline_only": True,
        }


PREDECLARED_VARIANTS: tuple[DixonColesVariantSpec, ...] = (
    DixonColesVariantSpec(
        model_name="champion_dc_xg",
        description="Frozen champion reference configuration.",
        config_overrides={},
    ),
    DixonColesVariantSpec(
        model_name="dc_rho_mild_minus_08",
        description="Less negative fixed Dixon-Coles rho (-0.08).",
        config_overrides={"rho": -0.08},
    ),
    DixonColesVariantSpec(
        model_name="dc_rho_stronger_minus_18",
        description="More negative fixed Dixon-Coles rho (-0.18).",
        config_overrides={"rho": -0.18},
    ),
    DixonColesVariantSpec(
        model_name="dc_fit_rho_each_fold",
        description="Fit rho inside each training fold using the existing champion grid search.",
        config_overrides={"rho": None},
        rho_behavior="fit_when_none",
    ),
    DixonColesVariantSpec(
        model_name="dc_score_grid_10",
        description="Extend the scoreline truncation grid from 8 to 10 goals.",
        config_overrides={"max_goals": 10},
    ),
    DixonColesVariantSpec(
        model_name="dc_alpha_030",
        description="Increase xG strength ridge regularisation from 0.15 to 0.30.",
        config_overrides={"alpha": 0.30},
    ),
    DixonColesVariantSpec(
        model_name="dc_decay_30d",
        description="Use a shorter 30-day time-decay half-life.",
        config_overrides={"decay_half_life_days": 30.0},
    ),
    DixonColesVariantSpec(
        model_name="dc_conservative_xg_shrinkage",
        description="Use larger np_xG pseudocount and stronger penalty-rate shrinkage.",
        config_overrides={"xg_pseudocount": 0.10, "penalty_shrinkage_n": 10.0},
    ),
)


def available_variant_names() -> list[str]:
    """Return predeclared variant names in evaluation order."""
    return [variant.model_name for variant in PREDECLARED_VARIANTS]


def get_variant_spec(model_name: str) -> DixonColesVariantSpec:
    """Return one predeclared variant spec or raise a clear error."""
    for variant in PREDECLARED_VARIANTS:
        if variant.model_name == model_name:
            return variant
    raise ValueError(f"Unknown Dixon-Coles variant {model_name!r}. Available variants: {available_variant_names()}")


def get_variant_provider(model_name: str) -> Callable[[], DixonColesVariantModel]:
    """Return a zero-argument provider compatible with rolling backtests."""
    spec_def = get_variant_spec(model_name)
    return lambda: DixonColesVariantModel(spec_def)


def build_model_config(spec_def: DixonColesVariantSpec) -> ModelConfig:
    """Build a production ModelConfig copy with variant overrides applied."""
    values = asdict(ModelConfig())
    values.update(spec_def.config_overrides)
    if isinstance(values["rho_grid"], list):
        values["rho_grid"] = tuple(values["rho_grid"])
    return ModelConfig(**values)


def _normalise_probability_columns(predictions: pd.DataFrame) -> None:
    """Convert champion percentage outputs to unit probabilities in-place."""
    probability_columns = ["p_home_win", "p_draw", "p_away_win"]
    if not set(probability_columns).issubset(predictions.columns):
        return
    row_sums = predictions[probability_columns].sum(axis=1)
    percentage_mask = row_sums > 1.5
    predictions.loc[percentage_mask, probability_columns] = (
        predictions.loc[percentage_mask, probability_columns] / 100.0
    )
    row_sums = predictions[probability_columns].sum(axis=1)
    valid_mask = row_sums > 0
    predictions.loc[valid_mask, probability_columns] = predictions.loc[valid_mask, probability_columns].div(
        row_sums.loc[valid_mask],
        axis=0,
    )


def run_dixon_coles_variant_experiment(
    df: pd.DataFrame,
    *,
    backtest_config: BacktestConfig,
    variant_names: tuple[str, ...] | None = None,
    n_bins: int = 5,
    top_n: int = 5,
) -> dict[str, Any]:
    """Run all requested variants on shared rolling folds."""
    names = variant_names or tuple(available_variant_names())
    from evaluation.backtesting import build_rolling_folds

    folds = build_rolling_folds(df, backtest_config)
    if not folds:
        raise ValueError("No backtest folds were created. Check dates and min_train_matches.")

    result_payloads = []
    prediction_frames = []
    for name in names:
        result = run_backtest_for_model(get_variant_provider(name), df, folds)
        predictions = attach_actual_results(result.predictions, df, folds)
        if not predictions.empty:
            prediction_frames.append(predictions)
        result_dict = result.to_dict()
        result_dict["predictions"] = predictions.to_dict(orient="records")
        result_payloads.append(result_dict)

    if not prediction_frames:
        raise ValueError("Dixon-Coles variants produced no predictions for the generated folds.")

    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    report_summary = build_report_summary(combined_predictions, n_bins=n_bins, top_n=top_n)
    markdown = render_variant_markdown(
        report_summary=report_summary,
        variants=[get_variant_spec(name) for name in names],
    )
    return {
        "backtest_config": config_to_dict(backtest_config),
        "folds": [fold.metadata() for fold in folds],
        "variants": [variant_payload(get_variant_spec(name)) for name in names],
        "model_results": result_payloads,
        "prediction_rows": combined_predictions.to_dict(orient="records"),
        "report_summary": report_summary,
        "markdown": markdown,
    }


def variant_payload(spec_def: DixonColesVariantSpec) -> dict[str, Any]:
    """Return a serializable variant definition."""
    return {
        "model_name": spec_def.model_name,
        "model_family": spec_def.model_family,
        "model_version": spec_def.model_version,
        "description": spec_def.description,
        "config_overrides": dict(spec_def.config_overrides),
        "rho_behavior": spec_def.rho_behavior,
    }


def render_variant_markdown(
    *,
    report_summary: dict[str, Any],
    variants: list[DixonColesVariantSpec],
) -> str:
    """Render a Dixon-Coles-specific report with the generic evaluation tables."""
    generic = render_markdown_report(report_summary)
    ranking = report_summary.get("comparison", [])
    champion = next((row for row in ranking if row["model_name"] == "champion_dc_xg"), None)
    best = ranking[0] if ranking else None
    summary_lines = [
        "# Dixon-Coles Configuration Variant Experiment",
        "",
        "## First-Run Summary",
        "",
        "This offline experiment keeps `champion_dc_xg` as the unchanged reference and tests a small",
        "predeclared grid of champion-family configuration variants.",
        "",
    ]
    if champion and best:
        if best["model_name"] == "champion_dc_xg":
            summary_lines.append(
                "No tested variant beat the champion reference on the primary ranking metric used by the report."
            )
        else:
            summary_lines.append(
                f"Best-ranked variant: `{best['model_name']}` versus champion reference `{champion['model_name']}`."
            )
        summary_lines.extend(
            [
                "",
                f"- Champion reference: Brier {champion['brier_score']:.4f}, "
                f"log loss {champion['log_loss']:.4f}, accuracy {champion['accuracy']:.4f}.",
                f"- Best ranked row: `{best['model_name']}` with Brier {best['brier_score']:.4f}, "
                f"log loss {best['log_loss']:.4f}, accuracy {best['accuracy']:.4f}.",
                "",
            ]
        )

    summary_lines.extend(
        [
            "## Predeclared Variants",
            "",
            "| model_name | change |",
            "| --- | --- |",
        ]
    )
    summary_lines.extend(f"| {variant.model_name} | {variant.description} |" for variant in variants)
    summary_lines.extend(["", generic.removeprefix("# WSL Model Evaluation Report\n\n")])
    return "\n".join(summary_lines)
