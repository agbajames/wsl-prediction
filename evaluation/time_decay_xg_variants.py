"""
Offline time-decay and xG-weighting champion-family variants.

These variants reuse the safe Dixon-Coles/champion-family wrapper and keep the
production champion implementation untouched. The grid is deliberately small
and predeclared to avoid broad hyperparameter search over one WSL season.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from evaluation.backtesting import BacktestConfig, build_rolling_folds, config_to_dict, run_backtest_for_model
from evaluation.dixon_coles_variants import (
    DixonColesVariantModel,
    DixonColesVariantSpec,
    render_variant_markdown,
    variant_payload,
)
from scripts.run_evaluation_report import build_report_summary
from scripts.run_model_comparison import attach_actual_results


PREDECLARED_TIME_DECAY_XG_VARIANTS: tuple[DixonColesVariantSpec, ...] = (
    DixonColesVariantSpec(
        model_name="champion_dc_xg",
        description="Frozen champion reference configuration: 60-day decay, fixed rho -0.13.",
        config_overrides={},
    ),
    DixonColesVariantSpec(
        model_name="dc_fit_rho_each_fold",
        description="Best Phase 8A-3 comparison reference: fit rho inside each training fold.",
        config_overrides={"rho": None},
        rho_behavior="fit_when_none",
    ),
    DixonColesVariantSpec(
        model_name="txg_decay_45d",
        description="Slightly shorter recency half-life: 45 days instead of 60.",
        config_overrides={"decay_half_life_days": 45.0},
    ),
    DixonColesVariantSpec(
        model_name="txg_decay_90d",
        description="Slightly longer recency half-life: 90 days instead of 60.",
        config_overrides={"decay_half_life_days": 90.0},
    ),
    DixonColesVariantSpec(
        model_name="txg_alpha_025",
        description="Moderately stronger xG team-strength ridge shrinkage: alpha 0.25.",
        config_overrides={"alpha": 0.25},
    ),
    DixonColesVariantSpec(
        model_name="txg_xg_pseudocount_010",
        description="Conservative np_xG floor/shrinkage: xG pseudocount 0.10.",
        config_overrides={"xg_pseudocount": 0.10},
    ),
    DixonColesVariantSpec(
        model_name="txg_conservative_weighting",
        description="Combined conservative xG weighting: alpha 0.25, xG pseudocount 0.10, penalty shrinkage 10.",
        config_overrides={"alpha": 0.25, "xg_pseudocount": 0.10, "penalty_shrinkage_n": 10.0},
    ),
)


def available_variant_names() -> list[str]:
    """Return predeclared Phase 8A-4 variant names in evaluation order."""
    return [variant.model_name for variant in PREDECLARED_TIME_DECAY_XG_VARIANTS]


def get_variant_spec(model_name: str) -> DixonColesVariantSpec:
    """Return one predeclared time-decay/xG variant spec or raise clearly."""
    for variant in PREDECLARED_TIME_DECAY_XG_VARIANTS:
        if variant.model_name == model_name:
            return variant
    raise ValueError(
        f"Unknown time-decay/xG variant {model_name!r}. Available variants: {available_variant_names()}"
    )


def get_variant_provider(model_name: str) -> Callable[[], DixonColesVariantModel]:
    """Return a zero-argument provider compatible with rolling backtests."""
    spec_def = get_variant_spec(model_name)
    return lambda: DixonColesVariantModel(spec_def)


def run_time_decay_xg_variant_experiment(
    df: pd.DataFrame,
    *,
    backtest_config: BacktestConfig,
    variant_names: tuple[str, ...] | None = None,
    n_bins: int = 5,
    top_n: int = 5,
) -> dict[str, Any]:
    """Run requested time-decay/xG variants on shared rolling folds."""
    names = variant_names or tuple(available_variant_names())
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
        raise ValueError("Time-decay/xG variants produced no predictions for the generated folds.")

    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    report_summary = build_report_summary(combined_predictions, n_bins=n_bins, top_n=top_n)
    markdown = render_time_decay_xg_markdown(
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


def render_time_decay_xg_markdown(
    *,
    report_summary: dict[str, Any],
    variants: list[DixonColesVariantSpec],
) -> str:
    """Render a Phase 8A-4 report with a roadmap-specific title."""
    markdown = render_variant_markdown(report_summary=report_summary, variants=variants)
    markdown = markdown.replace(
        "# Dixon-Coles Configuration Variant Experiment",
        "# Time-Decay and xG-Weighting Variant Experiment",
        1,
    )
    markdown = markdown.replace(
        "This offline experiment keeps `champion_dc_xg` as the unchanged reference and tests a small\n"
        "predeclared grid of champion-family configuration variants.",
        "This offline experiment keeps `champion_dc_xg` as the unchanged reference, includes\n"
        "`dc_fit_rho_each_fold` as the best Phase 8A-3 comparison reference, and tests a small\n"
        "predeclared grid of time-decay and xG-weighting variants.",
        1,
    )
    return markdown
