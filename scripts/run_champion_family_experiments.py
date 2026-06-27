#!/usr/bin/env python3
"""
Run consolidated champion-family calibration and decay experiments.

This runner is offline-only. It evaluates a small predeclared set of
champion-family variants, draw post-processing diagnostics, and fixed
champion-family blends on identical rolling folds. It does not modify or
promote the frozen champion implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.backtesting import BacktestConfig, BacktestFold, build_rolling_folds, config_to_dict
from evaluation.blending import BlendSpec, blend_prediction_frames, normalise_probability_frame
from evaluation.compare import PROBABILITY_COLUMNS, compare_model_results, summarize_model_results
from evaluation.dixon_coles_variants import DixonColesVariantModel, DixonColesVariantSpec, build_model_config
from evaluation.draw_adjustment import apply_draw_variant, draw_specific_metrics
from evaluation.metrics import outcome_indices, validate_probabilities
from scripts.run_model_comparison import load_match_csv

XG_SOURCE_COLUMNS = ("home_np_xg", "away_np_xg", "home_xg", "away_xg")
DEFAULT_TEST_START = "2025-10-01"
DEFAULT_TEST_END = "2026-05-16"
DEFAULT_MIN_TRAIN_MATCHES = 12
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_PREFIX = "champion_family_experiments"


CHAMPION_FAMILY_SPECS: tuple[DixonColesVariantSpec, ...] = (
    DixonColesVariantSpec(
        model_name="champion_dc_xg",
        description="Frozen champion reference: 60-day decay, fixed rho -0.13, xG pseudocount 0.05.",
        config_overrides={},
    ),
    DixonColesVariantSpec(
        model_name="dc_rho_mild_minus_08",
        description="Less negative fixed Dixon-Coles rho (-0.08).",
        config_overrides={"rho": -0.08},
    ),
    DixonColesVariantSpec(
        model_name="dc_fit_rho_each_fold",
        description="Fit rho inside each training fold using the existing champion grid search.",
        config_overrides={"rho": None},
        rho_behavior="fit_when_none",
    ),
    DixonColesVariantSpec(
        model_name="txg_xg_pseudocount_010",
        description="Conservative np_xG floor/shrinkage: xG pseudocount 0.10.",
        config_overrides={"xg_pseudocount": 0.10},
    ),
    DixonColesVariantSpec(
        model_name="txg_xg_pseudocount_015",
        description="Pseudocount sensitivity: xG pseudocount 0.15.",
        config_overrides={"xg_pseudocount": 0.15},
    ),
    DixonColesVariantSpec(
        model_name="txg_xg_pseudocount_020",
        description="Pseudocount sensitivity upper bound: xG pseudocount 0.20.",
        config_overrides={"xg_pseudocount": 0.20},
    ),
    DixonColesVariantSpec(
        model_name="txg_decay_75d",
        description="Conservative recency half-life sensitivity: 75 days.",
        config_overrides={"decay_half_life_days": 75.0},
    ),
    DixonColesVariantSpec(
        model_name="txg_decay_90d",
        description="Longer recency half-life diagnostic: 90 days.",
        config_overrides={"decay_half_life_days": 90.0},
    ),
    DixonColesVariantSpec(
        model_name="txg_decay_45d",
        description="Shorter recency half-life diagnostic holdover: 45 days.",
        config_overrides={"decay_half_life_days": 45.0},
    ),
)

CORE_CANDIDATES = (
    "champion_dc_xg",
    "dc_rho_mild_minus_08",
    "dc_fit_rho_each_fold",
    "txg_xg_pseudocount_010",
)
PSEUDOCOUNT_CANDIDATES = (
    "champion_dc_xg",
    "txg_xg_pseudocount_010",
    "txg_xg_pseudocount_015",
    "txg_xg_pseudocount_020",
)
DECAY_CANDIDATES = (
    "champion_dc_xg",
    "txg_decay_75d",
    "txg_decay_90d",
    "txg_decay_45d",
)

DRAW_VARIANTS = (
    {"variant_name": "additive_draw_-0.025", "method": "additive", "value": -0.025},
    {"variant_name": "multiplicative_draw_0.95", "method": "multiplicative", "value": 0.95},
    {"variant_name": "additive_draw_-0.050", "method": "additive", "value": -0.05},
)

CHAMPION_FAMILY_BLEND_SPECS: tuple[BlendSpec, ...] = (
    BlendSpec(
        model_name="blend_dc_fit_txg_50_50",
        components=("dc_fit_rho_each_fold", "txg_xg_pseudocount_010"),
        weights=(0.50, 0.50),
        description="Fixed blend of fitted-rho and xG-pseudocount champion-family candidates.",
    ),
    BlendSpec(
        model_name="blend_champion_dc_fit_50_50",
        components=("champion_dc_xg", "dc_fit_rho_each_fold"),
        weights=(0.50, 0.50),
        description="Fixed blend of frozen champion and fitted-rho candidate.",
    ),
    BlendSpec(
        model_name="blend_champion_txg_50_50",
        components=("champion_dc_xg", "txg_xg_pseudocount_010"),
        weights=(0.50, 0.50),
        description="Fixed blend of frozen champion and xG-pseudocount candidate.",
    ),
    BlendSpec(
        model_name="blend_champion_dc_fit_txg_34_33_33",
        components=("champion_dc_xg", "dc_fit_rho_each_fold", "txg_xg_pseudocount_010"),
        weights=(0.34, 0.33, 0.33),
        description="Fixed near-equal blend of the three champion-family references.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run consolidated champion-family experiments.")
    parser.add_argument("--csv", required=True, help="Local xG match-data CSV path.")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START, help="First test-window date, YYYY-MM-DD.")
    parser.add_argument("--test-end", default=DEFAULT_TEST_END, help="Final test-window date, YYYY-MM-DD.")
    parser.add_argument("--min-train-matches", type=int, default=DEFAULT_MIN_TRAIN_MATCHES)
    parser.add_argument("--test-window-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_match_csv(Path(args.csv))
    validate_xg_inputs(df)
    backtest_config = BacktestConfig(
        test_start=args.test_start,
        test_end=args.test_end,
        test_window_days=args.test_window_days,
        step_days=args.step_days,
        min_train_matches=args.min_train_matches,
    )
    payload = run_champion_family_experiments(
        df,
        csv_path=Path(args.csv),
        backtest_config=backtest_config,
    )
    write_outputs(payload, output_dir=Path(args.output_dir), prefix=args.prefix)


def validate_xg_inputs(df: pd.DataFrame) -> None:
    """Require real xG columns and usable xG values for completed rows."""
    missing = [column for column in XG_SOURCE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Champion-family xG experiments require xG columns. Missing: {missing}")
    completed = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()
    home_has_xg = completed[["home_np_xg", "home_xg"]].notna().any(axis=1)
    away_has_xg = completed[["away_np_xg", "away_xg"]].notna().any(axis=1)
    fallback_rows = int((~(home_has_xg & away_has_xg)).sum())
    if fallback_rows:
        raise ValueError(
            "Champion-family xG experiments require xG for both teams on every completed row. "
            f"Rows that would need fallback: {fallback_rows}."
        )


def run_champion_family_experiments(
    df: pd.DataFrame,
    *,
    csv_path: Path,
    backtest_config: BacktestConfig,
) -> dict[str, Any]:
    """Run predeclared champion-family candidates and return report payload."""
    folds = build_rolling_folds(df, backtest_config)
    if not folds:
        raise ValueError("No backtest folds were created. Check dates and min_train_matches.")

    prediction_frames: dict[str, pd.DataFrame] = {}
    rho_rows: list[dict[str, Any]] = []
    candidate_payloads: list[dict[str, Any]] = []

    for spec_def in CHAMPION_FAMILY_SPECS:
        predictions, diagnostics = run_variant_on_folds(df, folds=folds, spec_def=spec_def)
        prediction_frames[spec_def.model_name] = predictions
        rho_rows.extend(diagnostics)
        candidate_payloads.append(candidate_payload(spec_def))

    blend_frames = {}
    for blend_spec in CHAMPION_FAMILY_BLEND_SPECS:
        blend = blend_prediction_frames(prediction_frames, blend_spec)
        blend["candidate_category"] = "fixed_champion_family_blend"
        blend["candidate_description"] = blend_spec.description
        blend_frames[blend_spec.model_name] = blend
        candidate_payloads.append(blend_payload(blend_spec))

    draw_frames = build_draw_adjustment_frames(prediction_frames["champion_dc_xg"])
    all_predictions = pd.concat(
        [
            *prediction_frames.values(),
            *blend_frames.values(),
            *draw_frames.values(),
        ],
        ignore_index=True,
    )

    aggregate_metrics = build_aggregate_metrics(all_predictions)
    fold_metrics = build_fold_metrics(all_predictions, folds)
    rho_diagnostics = build_rho_diagnostics(pd.DataFrame(rho_rows), fold_metrics)
    draw_diagnostics = build_draw_diagnostics(all_predictions)
    observations = build_observations(aggregate_metrics, fold_metrics, rho_diagnostics, draw_diagnostics)
    markdown = render_markdown_report(
        csv_path=csv_path,
        backtest_config=backtest_config,
        folds=folds,
        aggregate_metrics=aggregate_metrics,
        rho_diagnostics=rho_diagnostics,
        draw_diagnostics=draw_diagnostics,
        observations=observations,
    )
    return {
        "csv_path": str(csv_path),
        "data_schema": data_schema_summary(df),
        "backtest_config": config_to_dict(backtest_config),
        "folds": [fold.metadata() for fold in folds],
        "candidates": candidate_payloads,
        "aggregate_metrics": aggregate_metrics.to_dict(orient="records"),
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "rho_diagnostics": rho_diagnostics.to_dict(orient="records"),
        "draw_diagnostics": draw_diagnostics.to_dict(orient="records"),
        "observations": observations,
        "markdown": markdown,
    }


def run_variant_on_folds(
    df: pd.DataFrame,
    *,
    folds: list[BacktestFold],
    spec_def: DixonColesVariantSpec,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run one champion-family variant manually so resolved rho can be captured."""
    frames = []
    diagnostics = []
    config = build_model_config(spec_def)
    rho_min, rho_max, _rho_step = config.rho_grid
    for fold in folds:
        train = df.loc[list(fold.train_indices)].copy()
        test = df.loc[list(fold.test_indices)].copy()
        model = DixonColesVariantModel(spec_def).fit(train)
        predictions = model.predict(test).copy()
        predictions["fold_id"] = fold.fold_id
        predictions["test_start"] = fold.test_start.date().isoformat()
        predictions["test_end"] = fold.test_end.date().isoformat()
        predictions["candidate_category"] = candidate_category(spec_def.model_name)
        predictions["candidate_description"] = spec_def.description
        predictions = attach_actuals(predictions, test)
        frames.append(predictions)

        resolved_rho = resolved_model_rho(model)
        diagnostics.append(
            {
                "model_name": spec_def.model_name,
                "fold_id": fold.fold_id,
                "test_start": fold.test_start.date().isoformat(),
                "test_end": fold.test_end.date().isoformat(),
                "train_size": fold.train_size,
                "test_size": fold.test_size,
                "resolved_rho": resolved_rho,
                "rho_grid_min": rho_min,
                "rho_grid_max": rho_max,
                "rho_hits_grid_min": bool(resolved_rho is not None and np.isclose(resolved_rho, rho_min)),
                "rho_hits_grid_max": bool(resolved_rho is not None and np.isclose(resolved_rho, rho_max)),
                "rho_behavior": spec_def.rho_behavior,
            }
        )
    return pd.concat(frames, ignore_index=True), diagnostics


def attach_actuals(predictions: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Attach actual outcomes to predictions in test row order."""
    actual = test.reset_index(drop=True).copy()
    out = normalise_probability_frame(predictions.reset_index(drop=True).copy())
    out["actual_outcome"] = [
        actual_outcome(row.home_goals, row.away_goals) for row in actual.itertuples(index=False)
    ]
    out["home_goals"] = actual["home_goals"].to_numpy()
    out["away_goals"] = actual["away_goals"].to_numpy()
    out["predicted_outcome"] = predicted_outcomes(out)
    return out


def build_draw_adjustment_frames(champion_predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build selected draw post-processing diagnostic rows from champion predictions."""
    probabilities = validate_probabilities(champion_predictions.loc[:, PROBABILITY_COLUMNS].astype(float).to_numpy())
    frames = {}
    for variant in DRAW_VARIANTS:
        adjusted_probabilities = apply_draw_variant(probabilities, variant)
        adjusted = champion_predictions.copy()
        adjusted.loc[:, PROBABILITY_COLUMNS] = adjusted_probabilities
        adjusted["model_name"] = variant["variant_name"]
        adjusted["model_family"] = "draw_post_processing_diagnostic"
        adjusted["model_version"] = "v1"
        adjusted["candidate_category"] = "draw_calibration_diagnostic"
        adjusted["candidate_description"] = f"{variant['method']} draw adjustment ({variant['value']})."
        adjusted["predicted_outcome"] = predicted_outcomes(adjusted)
        frames[variant["variant_name"]] = adjusted
    return frames


def build_aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate leaderboard with confidence diagnostics."""
    comparison = compare_model_results(predictions)
    extras = []
    for model_name, rows in predictions.groupby("model_name", sort=False):
        diagnostics = confidence_diagnostics(rows)
        category = single_value(rows, "candidate_category")
        description = single_value(rows, "candidate_description")
        n_folds = int(rows["fold_id"].nunique()) if "fold_id" in rows.columns else 0
        extras.append(
            {
                "model_name": model_name,
                "candidate_category": category,
                "candidate_description": description,
                "n_folds": n_folds,
                **diagnostics,
            }
        )
    aggregate = comparison.merge(pd.DataFrame(extras), on="model_name", how="left")
    return aggregate.sort_values(["log_loss", "brier_score", "accuracy"], ascending=[True, True, False]).reset_index(
        drop=True
    )


def build_fold_metrics(predictions: pd.DataFrame, folds: list[BacktestFold]) -> pd.DataFrame:
    """Return fold-level metrics and deltas versus champion for each candidate."""
    fold_lookup = {fold.fold_id: fold.metadata() for fold in folds}
    rows = []
    for (model_name, fold_id), group in predictions.groupby(["model_name", "fold_id"], sort=True):
        summary = summarize_model_results(group, model_name=model_name)
        metadata = fold_lookup[str(fold_id)]
        rows.append(
            {
                "model_name": model_name,
                "fold_id": fold_id,
                "test_start": metadata["test_start"],
                "test_end": metadata["test_end"],
                "train_size": metadata["train_size"],
                "test_size": metadata["test_size"],
                "log_loss": summary["log_loss"],
                "brier_score": summary["brier_score"],
                "accuracy": summary["accuracy"],
                "candidate_category": single_value(group, "candidate_category"),
            }
        )
    fold_metrics = pd.DataFrame(rows)
    champion = fold_metrics.loc[
        fold_metrics["model_name"] == "champion_dc_xg",
        ["fold_id", "log_loss", "brier_score", "accuracy"],
    ].rename(
        columns={
            "log_loss": "champion_log_loss",
            "brier_score": "champion_brier_score",
            "accuracy": "champion_accuracy",
        }
    )
    fold_metrics = fold_metrics.merge(champion, on="fold_id", how="left")
    fold_metrics["log_loss_delta_vs_champion"] = fold_metrics["log_loss"] - fold_metrics["champion_log_loss"]
    fold_metrics["brier_delta_vs_champion"] = fold_metrics["brier_score"] - fold_metrics["champion_brier_score"]
    fold_metrics["accuracy_delta_vs_champion"] = fold_metrics["accuracy"] - fold_metrics["champion_accuracy"]
    return fold_metrics.sort_values(["model_name", "fold_id"], kind="mergesort").reset_index(drop=True)


def build_rho_diagnostics(rho_rows: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach fold metrics and champion deltas to resolved-rho rows."""
    if rho_rows.empty:
        return rho_rows
    metric_columns = [
        "model_name",
        "fold_id",
        "log_loss",
        "brier_score",
        "accuracy",
        "log_loss_delta_vs_champion",
        "brier_delta_vs_champion",
        "accuracy_delta_vs_champion",
    ]
    return rho_rows.merge(fold_metrics.loc[:, metric_columns], on=["model_name", "fold_id"], how="left")


def build_draw_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return draw-specific metrics for all candidates."""
    rows = []
    for model_name, group in predictions.groupby("model_name", sort=False):
        probabilities = group.loc[:, PROBABILITY_COLUMNS].astype(float).to_numpy()
        outcomes = group["actual_outcome"].tolist()
        rows.append(
            {
                "model_name": model_name,
                "candidate_category": single_value(group, "candidate_category"),
                **draw_specific_metrics(probabilities, outcomes),
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate_category", "model_name"], kind="mergesort").reset_index(drop=True)


def confidence_diagnostics(rows: pd.DataFrame) -> dict[str, Any]:
    """Return aggregate overconfidence diagnostics."""
    probabilities = validate_probabilities(rows.loc[:, PROBABILITY_COLUMNS].astype(float).to_numpy())
    actual = outcome_indices(rows["actual_outcome"].tolist())
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    actual_probability = probabilities[np.arange(len(probabilities)), actual]
    return {
        "average_max_probability": float(confidence.mean()),
        "high_confidence_wrong_count": int(((confidence >= 0.8) & (predicted != actual)).sum()),
        "actual_probability_below_05_count": int((actual_probability < 0.05).sum()),
        "actual_probability_below_10_count": int((actual_probability < 0.10).sum()),
    }


def build_observations(
    aggregate_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    rho_diagnostics: pd.DataFrame,
    draw_diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    """Summarise key results for report rendering and JSON."""
    champion = find_row(aggregate_metrics, "champion_dc_xg")
    best_log = aggregate_metrics.iloc[0].to_dict()
    best_brier = aggregate_metrics.sort_values(["brier_score", "log_loss", "accuracy"], ascending=[True, True, False]).iloc[
        0
    ].to_dict()
    non_champion = aggregate_metrics.loc[aggregate_metrics["model_name"] != "champion_dc_xg"]
    beat_champion = non_champion.loc[
        (non_champion["log_loss"] < champion["log_loss"]) | (non_champion["brier_score"] < champion["brier_score"])
    ]
    rho_summary = summarise_rho(rho_diagnostics)
    return {
        "champion": round_record(champion),
        "best_by_log_loss": round_record(best_log),
        "best_by_brier": round_record(best_brier),
        "candidates_beating_champion": [round_record(row) for row in beat_champion.to_dict(orient="records")],
        "rho_summary": rho_summary,
        "pseudocount_summary": group_summary(aggregate_metrics, PSEUDOCOUNT_CANDIDATES),
        "decay_summary": group_summary(aggregate_metrics, DECAY_CANDIDATES),
        "blend_summary": group_summary(aggregate_metrics, tuple(spec.model_name for spec in CHAMPION_FAMILY_BLEND_SPECS)),
        "draw_summary": draw_group_summary(aggregate_metrics, draw_diagnostics),
        "fold_delta_summary": fold_delta_summary(fold_metrics),
    }


def summarise_rho(rho_diagnostics: pd.DataFrame) -> dict[str, Any]:
    fitted = rho_diagnostics.loc[rho_diagnostics["model_name"] == "dc_fit_rho_each_fold"].copy()
    if fitted.empty:
        return {}
    values = fitted["resolved_rho"].astype(float)
    improved_log = int((fitted["log_loss_delta_vs_champion"] < 0).sum())
    improved_brier = int((fitted["brier_delta_vs_champion"] < 0).sum())
    return {
        "fold_count": int(len(fitted)),
        "rho_min": float(values.min()),
        "rho_max": float(values.max()),
        "rho_mean": float(values.mean()),
        "rho_median": float(values.median()),
        "rho_std": float(values.std(ddof=0)),
        "grid_min_hits": int(fitted["rho_hits_grid_min"].sum()),
        "grid_max_hits": int(fitted["rho_hits_grid_max"].sum()),
        "folds_improved_log_loss_vs_champion": improved_log,
        "folds_improved_brier_vs_champion": improved_brier,
    }


def group_summary(aggregate_metrics: pd.DataFrame, model_names: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = aggregate_metrics.loc[aggregate_metrics["model_name"].isin(model_names)].copy()
    return [round_record(row) for row in rows.to_dict(orient="records")]


def draw_group_summary(aggregate_metrics: pd.DataFrame, draw_diagnostics: pd.DataFrame) -> list[dict[str, Any]]:
    names = tuple(variant["variant_name"] for variant in DRAW_VARIANTS)
    rows = aggregate_metrics.loc[aggregate_metrics["model_name"].isin(("champion_dc_xg", *names))].merge(
        draw_diagnostics,
        on=["model_name", "candidate_category"],
        how="left",
        suffixes=("", "_draw"),
    )
    return [round_record(row) for row in rows.to_dict(orient="records")]


def fold_delta_summary(fold_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for model_name, group in fold_metrics.groupby("model_name", sort=True):
        if model_name == "champion_dc_xg":
            continue
        rows.append(
            {
                "model_name": model_name,
                "folds_better_log_loss": int((group["log_loss_delta_vs_champion"] < 0).sum()),
                "folds_better_brier": int((group["brier_delta_vs_champion"] < 0).sum()),
                "mean_log_loss_delta_vs_champion": float(group["log_loss_delta_vs_champion"].mean()),
                "mean_brier_delta_vs_champion": float(group["brier_delta_vs_champion"].mean()),
            }
        )
    return [round_record(row) for row in rows]


def render_markdown_report(
    *,
    csv_path: Path,
    backtest_config: BacktestConfig,
    folds: list[BacktestFold],
    aggregate_metrics: pd.DataFrame,
    rho_diagnostics: pd.DataFrame,
    draw_diagnostics: pd.DataFrame,
    observations: dict[str, Any],
) -> str:
    """Render the consolidated champion-family report."""
    champion = observations["champion"]
    best_log = observations["best_by_log_loss"]
    best_brier = observations["best_by_brier"]
    rho_summary = observations["rho_summary"]
    lines = [
        "# Champion-Family Calibration and Decay Experiments",
        "",
        "## Executive Summary",
        "",
        (
            f"Best candidate by log loss: `{best_log['model_name']}` "
            f"(log loss {best_log['log_loss']:.4f}, Brier {best_log['brier_score']:.4f}, "
            f"accuracy {best_log['accuracy']:.4f})."
        ),
        (
            f"Best candidate by Brier score: `{best_brier['model_name']}` "
            f"(Brier {best_brier['brier_score']:.4f}, log loss {best_brier['log_loss']:.4f}, "
            f"accuracy {best_brier['accuracy']:.4f})."
        ),
        (
            f"Champion reference `champion_dc_xg`: log loss {champion['log_loss']:.4f}, "
            f"Brier {champion['brier_score']:.4f}, accuracy {champion['accuracy']:.4f}."
        ),
        "",
        "No variant should be promoted from this one offline window alone. Candidates that beat the champion",
        "remain shadow-ready hypotheses pending future-fixture validation.",
        "",
        "## Inputs and Protocol",
        "",
        f"- CSV used: `{csv_path.as_posix()}`",
        f"- Evaluation window: {pd.Timestamp(backtest_config.test_start).date()} to {pd.Timestamp(backtest_config.test_end).date()}",
        f"- Minimum training matches: {backtest_config.min_train_matches}",
        f"- Test window days: {backtest_config.test_window_days}",
        f"- Step days: {backtest_config.step_days}",
        f"- Rolling folds: {len(folds)}",
        f"- Evaluated fixtures per full candidate: {int(champion['n_matches'])}",
        "- xG columns required and verified: `home_np_xg`, `away_np_xg`, `home_xg`, `away_xg`.",
        "",
        "## Candidate List",
        "",
        _markdown_table(candidate_rows(), ["candidate_group", "model_name", "description"]),
        "",
        "## Aggregate Leaderboard",
        "",
        _markdown_table(
            aggregate_metrics.to_dict(orient="records"),
            [
                "rank",
                "model_name",
                "candidate_category",
                "n_matches",
                "n_folds",
                "log_loss",
                "brier_score",
                "accuracy",
                "average_max_probability",
                "high_confidence_wrong_count",
                "actual_probability_below_05_count",
                "actual_probability_below_10_count",
            ],
        ),
        "",
        "## Fold-Level Observations",
        "",
        _markdown_table(
            observations["fold_delta_summary"],
            [
                "model_name",
                "folds_better_log_loss",
                "folds_better_brier",
                "mean_log_loss_delta_vs_champion",
                "mean_brier_delta_vs_champion",
            ],
        ),
        "",
        "## Fitted-Rho Diagnostics",
        "",
    ]
    if rho_summary:
        lines.extend(
            [
                f"- Fitted-rho fold count: {rho_summary['fold_count']}",
                f"- Rho range: {rho_summary['rho_min']:.4f} to {rho_summary['rho_max']:.4f}",
                f"- Rho mean/median/std: {rho_summary['rho_mean']:.4f} / {rho_summary['rho_median']:.4f} / {rho_summary['rho_std']:.4f}",
                f"- Grid boundary hits: min={rho_summary['grid_min_hits']}, max={rho_summary['grid_max_hits']}",
                f"- Folds improved vs champion: log loss={rho_summary['folds_improved_log_loss_vs_champion']}, Brier={rho_summary['folds_improved_brier_vs_champion']}",
                "",
            ]
        )
    lines.extend(
        [
            _markdown_table(
                rho_diagnostics.loc[
                    rho_diagnostics["model_name"].isin(("champion_dc_xg", "dc_rho_mild_minus_08", "dc_fit_rho_each_fold"))
                ].to_dict(orient="records"),
                [
                    "model_name",
                    "fold_id",
                    "train_size",
                    "test_size",
                    "resolved_rho",
                    "rho_hits_grid_min",
                    "rho_hits_grid_max",
                    "log_loss_delta_vs_champion",
                    "brier_delta_vs_champion",
                ],
            ),
            "",
            "## Pseudocount Sensitivity",
            "",
            _markdown_table(
                observations["pseudocount_summary"],
                ["model_name", "log_loss", "brier_score", "accuracy", "average_max_probability"],
            ),
            "",
            "## Decay Sensitivity",
            "",
            _markdown_table(
                observations["decay_summary"],
                ["model_name", "log_loss", "brier_score", "accuracy", "average_max_probability"],
            ),
            "",
            "## Draw Calibration Diagnostics",
            "",
            _markdown_table(
                observations["draw_summary"],
                [
                    "model_name",
                    "log_loss",
                    "brier_score",
                    "accuracy",
                    "actual_draws",
                    "draw_prediction_rate",
                    "avg_predicted_draw_probability",
                    "draw_recall",
                    "draw_log_loss",
                    "non_draw_log_loss",
                    "non_draw_accuracy",
                ],
            ),
            "",
            "## Fixed Blend Results",
            "",
            _markdown_table(
                observations["blend_summary"],
                ["model_name", "log_loss", "brier_score", "accuracy", "average_max_probability"],
            ),
            "",
            "## Comparison Against Champion",
            "",
            _markdown_table(
                observations["candidates_beating_champion"],
                ["model_name", "candidate_category", "log_loss", "brier_score", "accuracy"],
            ),
            "",
            "## Recommendation",
            "",
            recommendation_text(observations),
            "",
            "## Guardrails",
            "",
            "- `champion_dc_xg` remains the frozen operational/reference model.",
            "- These candidates are separate offline variants, post-processing diagnostics, or fixed blends.",
            "- Market odds are not used as features or training inputs.",
            "- Neural-network work remains parked.",
            "- Do not promote a candidate from this one offline window alone.",
            "",
        ]
    )
    return "\n".join(lines)


def recommendation_text(observations: dict[str, Any]) -> str:
    best_log = observations["best_by_log_loss"]
    draw_summary = {row["model_name"]: row for row in observations.get("draw_summary", [])}
    best_draw = draw_summary.get(best_log["model_name"])
    rho_summary = observations.get("rho_summary", {})
    lines = [
        f"`{best_log['model_name']}` leads aggregate log loss/Brier in this offline run.",
        "Carry forward candidates that beat the champion on log loss or Brier as shadow-ready hypotheses only.",
    ]
    if best_draw is not None and best_draw.get("draw_recall", 1.0) <= 0:
        lines.append(
            f"`{best_log['model_name']}` also reduced draw recall to zero, so keep it as a draw-calibration "
            "diagnostic rather than a promotion-ready candidate."
        )
    if rho_summary:
        if rho_summary["grid_min_hits"] or rho_summary["grid_max_hits"]:
            lines.append("Fitted rho hit a grid boundary in at least one fold, so treat rho fitting as potentially noisy.")
        else:
            lines.append("Fitted rho did not hit the grid boundary, which supports continued shadow-style tracking.")
    lines.append("Prioritise future validation over additional offline tuning.")
    return "\n".join(f"- {line}" for line in lines)


def write_outputs(payload: dict[str, Any], *, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{prefix}_summary.md").write_text(payload["markdown"], encoding="utf-8")
    json_payload = {key: value for key, value in payload.items() if key != "markdown"}
    (output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(json_payload, default=json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(payload["fold_metrics"]).to_csv(output_dir / f"{prefix}_fold_metrics.csv", index=False)
    pd.DataFrame(payload["rho_diagnostics"]).to_csv(output_dir / f"{prefix}_rho_diagnostics.csv", index=False)
    pd.DataFrame(payload["draw_diagnostics"]).to_csv(output_dir / f"{prefix}_draw_diagnostics.csv", index=False)


def data_schema_summary(df: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(df["match_date"], errors="coerce")
    completed = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()
    return {
        "row_count": int(len(df)),
        "completed_row_count": int(len(completed)),
        "min_match_date": dates.min().date().isoformat() if pd.notna(dates.min()) else None,
        "max_match_date": dates.max().date().isoformat() if pd.notna(dates.max()) else None,
        "xg_columns_present": [column for column in XG_SOURCE_COLUMNS if column in df.columns],
        "xg_missingness": {column: int(df[column].isna().sum()) for column in XG_SOURCE_COLUMNS},
    }


def candidate_payload(spec_def: DixonColesVariantSpec) -> dict[str, Any]:
    config = asdict(build_model_config(spec_def))
    config["rho_grid"] = list(config["rho_grid"])
    return {
        "model_name": spec_def.model_name,
        "candidate_group": candidate_category(spec_def.model_name),
        "description": spec_def.description,
        "config_overrides": dict(spec_def.config_overrides),
        "rho_behavior": spec_def.rho_behavior,
        "config": config,
    }


def blend_payload(spec: BlendSpec) -> dict[str, Any]:
    return {
        "model_name": spec.model_name,
        "candidate_group": "fixed_champion_family_blend",
        "description": spec.description,
        "components": list(spec.components),
        "weights": list(spec.weights),
    }


def candidate_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "candidate_group": candidate_category(spec.model_name),
            "model_name": spec.model_name,
            "description": spec.description,
        }
        for spec in CHAMPION_FAMILY_SPECS
    ]
    rows.extend(
        {
            "candidate_group": "draw_calibration_diagnostic",
            "model_name": variant["variant_name"],
            "description": f"{variant['method']} draw adjustment ({variant['value']}).",
        }
        for variant in DRAW_VARIANTS
    )
    rows.extend(
        {
            "candidate_group": "fixed_champion_family_blend",
            "model_name": spec.model_name,
            "description": spec.description,
        }
        for spec in CHAMPION_FAMILY_BLEND_SPECS
    )
    return rows


def candidate_category(model_name: str) -> str:
    if model_name in CORE_CANDIDATES:
        return "core_champion_family"
    if model_name in PSEUDOCOUNT_CANDIDATES:
        return "xg_pseudocount_sensitivity"
    if model_name in DECAY_CANDIDATES:
        return "time_decay_sensitivity"
    return "champion_family_variant"


def resolved_model_rho(model: DixonColesVariantModel) -> float | None:
    champion = getattr(model, "_champion", None)
    rho = getattr(champion, "rho", None)
    if rho is None:
        return None
    return float(rho)


def actual_outcome(home_goals: Any, away_goals: Any) -> str:
    home = int(home_goals)
    away = int(away_goals)
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"


def predicted_outcomes(predictions: pd.DataFrame) -> list[str]:
    labels = {"p_home_win": "H", "p_draw": "D", "p_away_win": "A"}
    return [labels[column] for column in predictions.loc[:, PROBABILITY_COLUMNS].astype(float).idxmax(axis=1)]


def find_row(df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    rows = df.loc[df["model_name"] == model_name]
    if rows.empty:
        raise ValueError(f"Missing model row: {model_name}")
    return rows.iloc[0].to_dict()


def single_value(rows: pd.DataFrame, column: str) -> Any:
    if column not in rows.columns:
        return ""
    values = rows[column].dropna().astype(str).unique().tolist()
    return values[0] if values else ""


def round_record(row: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in row.items():
        if isinstance(value, float):
            result[key] = round(value, 6)
        else:
            result[key] = value
    return result


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    if not rows:
        return "\n".join([header, divider])
    body = ["| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def format_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


if __name__ == "__main__":
    main()
