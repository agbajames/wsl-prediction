"""
Offline fixed-weight non-market blending experiments.

The blends in this module consume existing model probability outputs from the
shared rolling backtest framework. They do not change production prediction
behaviour and intentionally use a small predeclared grid of weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from evaluation.backtesting import BacktestConfig, build_rolling_folds, config_to_dict, run_backtest_for_model
from evaluation.compare import PROBABILITY_COLUMNS
from evaluation.dixon_coles_variants import get_variant_provider as get_dixon_coles_variant_provider
from evaluation.time_decay_xg_variants import get_variant_provider as get_time_decay_xg_variant_provider
from experiments.registry import get_model_constructor
from models.base import EvaluationModel
from scripts.run_evaluation_report import build_report_summary, render_markdown_report
from scripts.run_model_comparison import attach_actual_results


@dataclass(frozen=True)
class BlendSpec:
    """Serializable definition of one fixed-weight offline blend."""

    model_name: str
    components: tuple[str, ...]
    weights: tuple[float, ...]
    description: str

    @property
    def model_family(self) -> str:
        return "fixed_weight_non_market_blend"

    @property
    def model_version(self) -> str:
        return "v1"


PREDECLARED_BLEND_SPECS: tuple[BlendSpec, ...] = (
    BlendSpec(
        model_name="blend_champion_regularised_50_50",
        components=("champion_dc_xg", "regularised_team_strength"),
        weights=(0.50, 0.50),
        description="Even fixed blend of champion Dixon-Coles/xG and regularised team strength.",
    ),
    BlendSpec(
        model_name="blend_champion_regularised_70_30",
        components=("champion_dc_xg", "regularised_team_strength"),
        weights=(0.70, 0.30),
        description="Champion-led blend with the strongest standalone statistical challenger.",
    ),
    BlendSpec(
        model_name="blend_champion_improved_logistic_70_30",
        components=("champion_dc_xg", "improved_logistic_regression"),
        weights=(0.70, 0.30),
        description="Champion-led blend with the Phase 8B improved logistic challenger.",
    ),
    BlendSpec(
        model_name="blend_champion_random_forest_70_30",
        components=("champion_dc_xg", "random_forest"),
        weights=(0.70, 0.30),
        description="Champion-led blend with the Phase 8B random forest challenger.",
    ),
    BlendSpec(
        model_name="blend_champion_regularised_improved_logistic_60_20_20",
        components=("champion_dc_xg", "regularised_team_strength", "improved_logistic_regression"),
        weights=(0.60, 0.20, 0.20),
        description="Three-model blend anchored on champion with two non-market challengers.",
    ),
    BlendSpec(
        model_name="blend_dc_fit_txg_50_50",
        components=("dc_fit_rho_each_fold", "txg_xg_pseudocount_010"),
        weights=(0.50, 0.50),
        description="Small Phase 8A champion-family reference blend of fitted-rho and conservative xG shrinkage.",
    ),
)


DEFAULT_COMPONENT_NAMES: tuple[str, ...] = (
    "champion_dc_xg",
    "dc_fit_rho_each_fold",
    "txg_xg_pseudocount_010",
    "regularised_team_strength",
    "improved_logistic_regression",
    "random_forest",
    "neural_network",
)


ModelProvider = Callable[[], EvaluationModel]


def available_blend_names() -> list[str]:
    """Return predeclared blend names in evaluation order."""
    return [spec.model_name for spec in PREDECLARED_BLEND_SPECS]


def get_blend_spec(model_name: str) -> BlendSpec:
    """Return one predeclared blend spec or raise clearly."""
    for spec in PREDECLARED_BLEND_SPECS:
        if spec.model_name == model_name:
            return spec
    raise ValueError(f"Unknown blend {model_name!r}. Available blends: {available_blend_names()}")


def build_default_component_providers() -> dict[str, ModelProvider]:
    """Build providers for the default Phase 8D-1 component set."""
    return {
        "champion_dc_xg": get_model_constructor("champion_dc_xg"),
        "dc_fit_rho_each_fold": get_dixon_coles_variant_provider("dc_fit_rho_each_fold"),
        "txg_xg_pseudocount_010": get_time_decay_xg_variant_provider("txg_xg_pseudocount_010"),
        "regularised_team_strength": get_model_constructor("regularised_team_strength"),
        "improved_logistic_regression": get_model_constructor("improved_logistic_regression"),
        "random_forest": get_model_constructor("random_forest"),
        "neural_network": get_model_constructor("neural_network"),
    }


def normalise_probability_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with H/D/A columns in unit-probability scale."""
    missing = [column for column in PROBABILITY_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")

    normalized = predictions.copy()
    probabilities = normalized.loc[:, PROBABILITY_COLUMNS].astype(float)
    if (probabilities < 0).any().any():
        raise ValueError("Probability columns cannot contain negative values.")

    row_sums = probabilities.sum(axis=1)
    percentage_mask = row_sums > 1.5
    probabilities.loc[percentage_mask] = probabilities.loc[percentage_mask] / 100.0
    row_sums = probabilities.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("Probability rows must have a positive sum.")

    normalized.loc[:, PROBABILITY_COLUMNS] = probabilities.div(row_sums, axis=0)
    return normalized


def blend_prediction_frames(
    predictions_by_model: dict[str, pd.DataFrame],
    spec: BlendSpec,
) -> pd.DataFrame:
    """Blend aligned prediction frames according to one fixed-weight spec."""
    _validate_blend_spec(spec)
    missing = [component for component in spec.components if component not in predictions_by_model]
    if missing:
        raise ValueError(f"Blend {spec.model_name!r} missing component predictions: {missing}")

    weights = _normalised_weights(spec.weights)
    frames = [normalise_probability_frame(predictions_by_model[component]) for component in spec.components]
    _validate_aligned_frames(spec, frames)

    template = frames[0].copy()
    blended = sum(weight * frame.loc[:, PROBABILITY_COLUMNS].astype(float) for weight, frame in zip(weights, frames))
    blended = blended.div(blended.sum(axis=1), axis=0)
    template.loc[:, PROBABILITY_COLUMNS] = blended
    template["model_name"] = spec.model_name
    template["model_family"] = spec.model_family
    template["model_version"] = spec.model_version
    template["blend_components"] = ", ".join(spec.components)
    template["blend_weights"] = ", ".join(f"{weight:.4f}" for weight in weights)
    template["blend_description"] = spec.description
    template["predicted_outcome"] = _predicted_outcomes(template)
    return template


def run_non_market_blending_experiment(
    df: pd.DataFrame,
    *,
    backtest_config: BacktestConfig,
    component_providers: dict[str, ModelProvider] | None = None,
    blend_specs: tuple[BlendSpec, ...] | None = None,
    n_bins: int = 5,
    top_n: int = 5,
) -> dict[str, Any]:
    """Run components and fixed-weight blends on shared rolling folds."""
    providers = component_providers or build_default_component_providers()
    specs = blend_specs or PREDECLARED_BLEND_SPECS
    folds = build_rolling_folds(df, backtest_config)
    if not folds:
        raise ValueError("No backtest folds were created. Check dates and min_train_matches.")

    result_payloads = []
    prediction_frames_by_model: dict[str, pd.DataFrame] = {}
    for model_name, provider in providers.items():
        result = run_backtest_for_model(provider, df, folds)
        predictions = attach_actual_results(result.predictions, df, folds)
        predictions = normalise_probability_frame(predictions) if not predictions.empty else predictions
        if not predictions.empty:
            prediction_frames_by_model[model_name] = predictions
        result_dict = result.to_dict()
        result_dict["predictions"] = predictions.to_dict(orient="records")
        result_payloads.append(result_dict)

    if not prediction_frames_by_model:
        raise ValueError("Component models produced no predictions for the generated folds.")

    blend_frames = [blend_prediction_frames(prediction_frames_by_model, spec) for spec in specs]
    combined_predictions = pd.concat([*prediction_frames_by_model.values(), *blend_frames], ignore_index=True)
    report_summary = build_report_summary(combined_predictions, n_bins=n_bins, top_n=top_n)
    markdown = render_blending_markdown(
        report_summary=report_summary,
        blend_specs=list(specs),
        component_names=tuple(providers),
    )
    return {
        "backtest_config": config_to_dict(backtest_config),
        "folds": [fold.metadata() for fold in folds],
        "components": list(providers),
        "blends": [blend_payload(spec) for spec in specs],
        "model_results": result_payloads,
        "prediction_rows": combined_predictions.to_dict(orient="records"),
        "report_summary": report_summary,
        "markdown": markdown,
    }


def blend_payload(spec: BlendSpec) -> dict[str, Any]:
    """Return a serializable blend definition."""
    return {
        "model_name": spec.model_name,
        "model_family": spec.model_family,
        "model_version": spec.model_version,
        "components": list(spec.components),
        "weights": list(spec.weights),
        "description": spec.description,
        "offline_only": True,
    }


def render_blending_markdown(
    *,
    report_summary: dict[str, Any],
    blend_specs: list[BlendSpec],
    component_names: tuple[str, ...],
) -> str:
    """Render a Phase 8D-1 report with the generic evaluation tables."""
    generic = render_markdown_report(report_summary)
    ranking = report_summary.get("comparison", [])
    champion = next((row for row in ranking if row["model_name"] == "champion_dc_xg"), None)
    blend_rows = [row for row in ranking if row["model_name"].startswith("blend_")]
    best = ranking[0] if ranking else None
    best_blend = min(blend_rows, key=lambda row: (row["log_loss"], row["brier_score"], -row["accuracy"]), default=None)

    lines = [
        "# Non-Market Blending Experiment",
        "",
        "## First-Run Summary",
        "",
        "This offline experiment keeps `champion_dc_xg` as the unchanged operational reference and tests",
        "a small predeclared set of fixed-weight blends over existing non-market model probabilities.",
        "",
    ]
    if champion and best:
        lines.extend(
            [
                f"- Champion reference: Brier {champion['brier_score']:.4f}, "
                f"log loss {champion['log_loss']:.4f}, accuracy {champion['accuracy']:.4f}.",
                f"- Best ranked row: `{best['model_name']}` with Brier {best['brier_score']:.4f}, "
                f"log loss {best['log_loss']:.4f}, accuracy {best['accuracy']:.4f}.",
            ]
        )
    if best_blend:
        lines.append(
            f"- Best blend: `{best_blend['model_name']}` with Brier {best_blend['brier_score']:.4f}, "
            f"log loss {best_blend['log_loss']:.4f}, accuracy {best_blend['accuracy']:.4f}."
        )
    lines.extend(
        [
            "",
            "No production promotion should happen from this PR alone; these blends are evaluation-only",
            "signals for future investigation.",
            "",
            "## Components",
            "",
            ", ".join(f"`{name}`" for name in component_names),
            "",
            "## Predeclared Blends",
            "",
            "| model_name | components | weights | change |",
            "| --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        "| "
        + " | ".join(
            [
                spec.model_name,
                ", ".join(spec.components),
                ", ".join(f"{weight:.2f}" for weight in spec.weights),
                spec.description,
            ]
        )
        + " |"
        for spec in blend_specs
    )
    lines.extend(["", generic.removeprefix("# WSL Model Evaluation Report\n\n")])
    return "\n".join(lines)


def _validate_blend_spec(spec: BlendSpec) -> None:
    if len(spec.components) != len(spec.weights):
        raise ValueError(f"Blend {spec.model_name!r} must have one weight per component.")
    if len(spec.components) < 2:
        raise ValueError(f"Blend {spec.model_name!r} must have at least two components.")
    if any(weight < 0 for weight in spec.weights):
        raise ValueError(f"Blend {spec.model_name!r} cannot contain negative weights.")
    if sum(spec.weights) <= 0:
        raise ValueError(f"Blend {spec.model_name!r} must have a positive total weight.")


def _normalised_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
    total = float(sum(weights))
    return tuple(float(weight) / total for weight in weights)


def _validate_aligned_frames(spec: BlendSpec, frames: list[pd.DataFrame]) -> None:
    expected_length = len(frames[0])
    key_columns = [column for column in ("fold_id", "home_team", "away_team", "match_date") if column in frames[0].columns]
    for frame in frames:
        if len(frame) != expected_length:
            raise ValueError(f"Blend {spec.model_name!r} component frames have different lengths.")
        missing_keys = [column for column in key_columns if column not in frame.columns]
        if missing_keys:
            raise ValueError(f"Blend {spec.model_name!r} component frame missing alignment columns: {missing_keys}")
        for column in key_columns:
            left = frames[0][column].astype(str).reset_index(drop=True)
            right = frame[column].astype(str).reset_index(drop=True)
            if not left.equals(right):
                raise ValueError(f"Blend {spec.model_name!r} component frames are not aligned on {column!r}.")


def _predicted_outcomes(predictions: pd.DataFrame) -> list[str]:
    labels = {"p_home_win": "H", "p_draw": "D", "p_away_win": "A"}
    return [labels[column] for column in predictions.loc[:, PROBABILITY_COLUMNS].astype(float).idxmax(axis=1)]
