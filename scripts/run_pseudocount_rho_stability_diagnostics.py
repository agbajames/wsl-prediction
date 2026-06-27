#!/usr/bin/env python3
"""Run focused pseudocount and fitted-rho stability diagnostics."""

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
from evaluation.compare import PROBABILITY_COLUMNS, compare_model_results, summarize_model_results
from evaluation.dixon_coles_variants import DixonColesVariantModel, DixonColesVariantSpec, build_model_config
from evaluation.metrics import outcome_indices, validate_probabilities
from scripts.run_champion_family_experiments import (
    attach_actuals,
    data_schema_summary,
    format_cell,
    json_default,
    predicted_outcomes,
    validate_xg_inputs,
)
from scripts.run_model_comparison import load_match_csv

DEFAULT_CSV = "data/exports/wsl_match_data_xg_phase3_20260626_fresh.csv"
DEFAULT_TEST_START = "2025-10-01"
DEFAULT_TEST_END = "2026-05-16"
DEFAULT_MIN_TRAIN_MATCHES = 12
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_PREFIX = "pseudocount_rho_stability_diagnostics"


PSEUDOCOUNT_SPECS: tuple[DixonColesVariantSpec, ...] = (
    DixonColesVariantSpec(
        model_name="champion_dc_xg",
        description="Champion/default pseudocount 0.05 and fixed rho -0.13.",
        config_overrides={},
    ),
    DixonColesVariantSpec(
        model_name="txg_xg_pseudocount_010",
        description="xG pseudocount 0.10.",
        config_overrides={"xg_pseudocount": 0.10},
    ),
    DixonColesVariantSpec(
        model_name="txg_xg_pseudocount_015",
        description="xG pseudocount 0.15.",
        config_overrides={"xg_pseudocount": 0.15},
    ),
    DixonColesVariantSpec(
        model_name="txg_xg_pseudocount_020",
        description="xG pseudocount 0.20.",
        config_overrides={"xg_pseudocount": 0.20},
    ),
    DixonColesVariantSpec(
        model_name="txg_xg_pseudocount_025",
        description="xG pseudocount 0.25 saturation diagnostic.",
        config_overrides={"xg_pseudocount": 0.25},
    ),
)

RHO_SPECS: tuple[DixonColesVariantSpec, ...] = (
    DixonColesVariantSpec(
        model_name="dc_rho_mild_minus_08",
        description="Fixed rho -0.08 comparator.",
        config_overrides={"rho": -0.08},
    ),
    DixonColesVariantSpec(
        model_name="dc_fit_rho_each_fold",
        description="Fit rho inside each training fold using current champion grid.",
        config_overrides={"rho": None},
        rho_behavior="fit_when_none",
    ),
    DixonColesVariantSpec(
        model_name="dc_fit_rho_each_fold_wide_grid",
        description="Diagnostic-only fitted rho with wider positive grid bound.",
        config_overrides={"rho": None, "rho_grid": (-0.30, 0.10, 0.01)},
        rho_behavior="fit_when_none",
    ),
)

PSEUDOCOUNT_VALUES = {
    "champion_dc_xg": 0.05,
    "txg_xg_pseudocount_010": 0.10,
    "txg_xg_pseudocount_015": 0.15,
    "txg_xg_pseudocount_020": 0.20,
    "txg_xg_pseudocount_025": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused pseudocount/rho stability diagnostics.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Local verified xG CSV path.")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default=DEFAULT_TEST_END)
    parser.add_argument("--min-train-matches", type=int, default=DEFAULT_MIN_TRAIN_MATCHES)
    parser.add_argument("--test-window-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Verified xG CSV not found: {csv_path}")

    df = load_match_csv(csv_path)
    validate_xg_inputs(df)
    backtest_config = BacktestConfig(
        test_start=args.test_start,
        test_end=args.test_end,
        test_window_days=args.test_window_days,
        step_days=args.step_days,
        min_train_matches=args.min_train_matches,
    )
    payload = run_diagnostics(df, csv_path=csv_path, backtest_config=backtest_config)
    write_outputs(payload, output_dir=Path(args.output_dir), prefix=args.prefix)


def run_diagnostics(
    df: pd.DataFrame,
    *,
    csv_path: Path,
    backtest_config: BacktestConfig,
) -> dict[str, Any]:
    folds = build_rolling_folds(df, backtest_config)
    if not folds:
        raise ValueError("No backtest folds were created. Check dates and min_train_matches.")

    prediction_frames: dict[str, pd.DataFrame] = {}
    rho_rows: list[dict[str, Any]] = []
    candidates = list(PSEUDOCOUNT_SPECS) + list(RHO_SPECS)

    for spec_def in candidates:
        predictions, diagnostics = run_variant_on_folds(df, folds=folds, spec_def=spec_def)
        prediction_frames[spec_def.model_name] = predictions
        rho_rows.extend(diagnostics)

    combined_spec = build_combined_candidate(prediction_frames, folds)
    if combined_spec is not None:
        predictions, diagnostics = run_variant_on_folds(df, folds=folds, spec_def=combined_spec)
        prediction_frames[combined_spec.model_name] = predictions
        rho_rows.extend(diagnostics)
        candidates.append(combined_spec)

    all_predictions = pd.concat(prediction_frames.values(), ignore_index=True)
    aggregate_metrics = build_aggregate_metrics(all_predictions)
    fold_metrics = build_fold_metrics(all_predictions, folds)
    rho_diagnostics = build_rho_diagnostics(pd.DataFrame(rho_rows), fold_metrics)
    pseudocount_summary = build_pseudocount_summary(aggregate_metrics, fold_metrics, all_predictions)
    rho_summary = build_rho_summary(aggregate_metrics, fold_metrics, rho_diagnostics)
    combined_summary = build_combined_summary(aggregate_metrics, fold_metrics, combined_spec)
    observations = build_observations(
        aggregate_metrics=aggregate_metrics,
        pseudocount_summary=pseudocount_summary,
        rho_summary=rho_summary,
        combined_summary=combined_summary,
    )

    payload = {
        "csv_path": str(csv_path),
        "data_schema": data_schema_summary(df),
        "backtest_config": config_to_dict(backtest_config),
        "folds": [fold.metadata() for fold in folds],
        "candidates": [candidate_payload(spec_def) for spec_def in candidates],
        "aggregate_metrics": aggregate_metrics.to_dict(orient="records"),
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "rho_diagnostics": rho_diagnostics.to_dict(orient="records"),
        "pseudocount_summary": pseudocount_summary,
        "rho_summary": rho_summary,
        "combined_summary": combined_summary,
        "observations": observations,
    }
    payload["markdown"] = render_markdown_report(payload)
    return payload


def run_variant_on_folds(
    df: pd.DataFrame,
    *,
    folds: list[BacktestFold],
    spec_def: DixonColesVariantSpec,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    diagnostics = []
    config = build_model_config(spec_def)
    rho_grid_min = float(config.rho_grid[0])
    rho_grid_max = float(config.rho_grid[1])

    for fold in folds:
        train = df.loc[list(fold.train_indices)].copy()
        test = df.loc[list(fold.test_indices)].copy()
        model = DixonColesVariantModel(spec_def).fit(train)
        predictions = attach_actuals(model.predict(test).copy(), test)
        predictions["fold_id"] = fold.fold_id
        predictions["test_start"] = fold.test_start.date().isoformat()
        predictions["test_end"] = fold.test_end.date().isoformat()
        predictions["candidate_category"] = candidate_category(spec_def.model_name)
        predictions["candidate_description"] = spec_def.description
        predictions["predicted_outcome"] = predicted_outcomes(predictions)
        frames.append(predictions)

        rho = resolved_model_rho(model)
        diagnostics.append(
            {
                "model_name": spec_def.model_name,
                "fold_id": fold.fold_id,
                "train_size": fold.train_size,
                "test_size": fold.test_size,
                "test_start": fold.test_start.date().isoformat(),
                "test_end": fold.test_end.date().isoformat(),
                "resolved_rho": rho,
                "rho_grid_min": rho_grid_min,
                "rho_grid_max": rho_grid_max,
                "rho_hits_grid_min": bool(rho is not None and np.isclose(rho, rho_grid_min)),
                "rho_hits_grid_max": bool(rho is not None and np.isclose(rho, rho_grid_max)),
                "rho_behavior": spec_def.rho_behavior,
            }
        )

    return pd.concat(frames, ignore_index=True), diagnostics


def build_combined_candidate(
    prediction_frames: dict[str, pd.DataFrame],
    folds: list[BacktestFold],
) -> DixonColesVariantSpec | None:
    del folds
    baseline = prediction_frames["champion_dc_xg"]
    candidates = []
    for model_name in ("txg_xg_pseudocount_020", "txg_xg_pseudocount_025"):
        metrics = candidate_vs_baseline(prediction_frames[model_name], baseline)
        if metrics["aggregate_better_log_loss"] and metrics["folds_better_log_loss"] >= 10:
            candidates.append((model_name, metrics))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1]["log_loss"], item[1]["brier_score"]))
    best_name = candidates[0][0]
    value = PSEUDOCOUNT_VALUES[best_name]
    suffix = f"{int(round(value * 100)):03d}"
    return DixonColesVariantSpec(
        model_name=f"dc_fit_rho_xg_pseudocount_{suffix}",
        description=f"Diagnostic-only fitted rho plus xG pseudocount {value:.2f}.",
        config_overrides={"rho": None, "xg_pseudocount": value},
        rho_behavior="fit_when_none",
    )


def candidate_vs_baseline(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    candidate_summary = summarize_model_results(candidate, model_name=str(candidate["model_name"].iloc[0]))
    baseline_summary = summarize_model_results(baseline, model_name=str(baseline["model_name"].iloc[0]))
    fold_rows = []
    for fold_id, fold_candidate in candidate.groupby("fold_id", sort=True):
        fold_baseline = baseline[baseline["fold_id"] == fold_id]
        if fold_baseline.empty:
            continue
        candidate_fold = summarize_model_results(fold_candidate, model_name=str(candidate["model_name"].iloc[0]))
        baseline_fold = summarize_model_results(fold_baseline, model_name=str(baseline["model_name"].iloc[0]))
        fold_rows.append(
            {
                "log_loss_delta_vs_baseline": candidate_fold["log_loss"] - baseline_fold["log_loss"],
                "brier_delta_vs_baseline": candidate_fold["brier_score"] - baseline_fold["brier_score"],
            }
        )
    return {
        "log_loss": candidate_summary["log_loss"],
        "brier_score": candidate_summary["brier_score"],
        "aggregate_better_log_loss": candidate_summary["log_loss"] < baseline_summary["log_loss"],
        "aggregate_better_brier": candidate_summary["brier_score"] < baseline_summary["brier_score"],
        "folds_better_log_loss": sum(row["log_loss_delta_vs_baseline"] < 0 for row in fold_rows),
        "folds_better_brier": sum(row["brier_delta_vs_baseline"] < 0 for row in fold_rows),
    }


def build_aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    aggregate = compare_model_results(predictions)
    extras = []
    for model_name, rows in predictions.groupby("model_name", sort=True):
        row = {
            "model_name": model_name,
            "candidate_category": single_value(rows, "candidate_category"),
            "candidate_description": single_value(rows, "candidate_description"),
            "n_folds": int(rows["fold_id"].nunique()),
            **confidence_diagnostics(rows),
            **draw_diagnostics(rows),
            **favourite_diagnostics(rows),
        }
        extras.append(row)
    aggregate = aggregate.merge(pd.DataFrame(extras), on="model_name", how="left")
    aggregate = aggregate.sort_values(
        ["log_loss", "brier_score", "accuracy", "model_name"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    aggregate["rank"] = aggregate.index + 1
    return aggregate


def build_fold_metrics(predictions: pd.DataFrame, folds: list[BacktestFold]) -> pd.DataFrame:
    fold_meta = {fold.fold_id: fold.metadata() for fold in folds}
    rows = []
    for (model_name, fold_id), group in predictions.groupby(["model_name", "fold_id"], sort=True):
        summary = summarize_model_results(group, model_name=str(model_name))
        meta = fold_meta[str(fold_id)]
        rows.append(
            {
                **summary,
                "fold_id": fold_id,
                "test_start": meta["test_start"],
                "test_end": meta["test_end"],
                "train_size": meta["train_size"],
                "test_size": meta["test_size"],
                "candidate_category": single_value(group, "candidate_category"),
            }
        )
    fold_metrics = pd.DataFrame(rows)
    baseline = fold_metrics[fold_metrics["model_name"] == "champion_dc_xg"][
        ["fold_id", "log_loss", "brier_score", "accuracy"]
    ].rename(
        columns={
            "log_loss": "champion_log_loss",
            "brier_score": "champion_brier_score",
            "accuracy": "champion_accuracy",
        }
    )
    fold_metrics = fold_metrics.merge(baseline, on="fold_id", how="left")
    fold_metrics["log_loss_delta_vs_champion"] = fold_metrics["log_loss"] - fold_metrics["champion_log_loss"]
    fold_metrics["brier_delta_vs_champion"] = fold_metrics["brier_score"] - fold_metrics["champion_brier_score"]
    fold_metrics["accuracy_delta_vs_champion"] = fold_metrics["accuracy"] - fold_metrics["champion_accuracy"]
    return fold_metrics.sort_values(["model_name", "fold_id"]).reset_index(drop=True)


def build_rho_diagnostics(rho_rows: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    merged = rho_rows.merge(
        fold_metrics[
            [
                "model_name",
                "fold_id",
                "log_loss",
                "brier_score",
                "accuracy",
                "log_loss_delta_vs_champion",
                "brier_delta_vs_champion",
                "accuracy_delta_vs_champion",
            ]
        ],
        on=["model_name", "fold_id"],
        how="left",
    )
    return merged.sort_values(["model_name", "fold_id"]).reset_index(drop=True)


def build_pseudocount_summary(
    aggregate_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    champion_draw_log_loss = float(
        aggregate_metrics.loc[aggregate_metrics["model_name"] == "champion_dc_xg", "draw_log_loss"].iloc[0]
    )
    for model_name, pseudocount in PSEUDOCOUNT_VALUES.items():
        aggregate = find_row(aggregate_metrics, model_name)
        folds = fold_metrics[fold_metrics["model_name"] == model_name]
        model_predictions = predictions[predictions["model_name"] == model_name]
        rows.append(
            round_record(
                {
                    **aggregate,
                    "xg_pseudocount": pseudocount,
                    "folds_better_log_loss_vs_005": int((folds["log_loss_delta_vs_champion"] < 0).sum()),
                    "folds_worse_log_loss_vs_005": int((folds["log_loss_delta_vs_champion"] > 0).sum()),
                    "folds_better_brier_vs_005": int((folds["brier_delta_vs_champion"] < 0).sum()),
                    "folds_worse_brier_vs_005": int((folds["brier_delta_vs_champion"] > 0).sum()),
                    "mean_log_loss_delta_vs_005": float(folds["log_loss_delta_vs_champion"].mean()),
                    "mean_brier_delta_vs_005": float(folds["brier_delta_vs_champion"].mean()),
                    "draw_log_loss_delta_vs_005": float(aggregate["draw_log_loss"] - champion_draw_log_loss),
                    "favourite_breakdown": favourite_breakdown(model_predictions),
                }
            )
        )
    return sorted(rows, key=lambda row: row["xg_pseudocount"])


def build_rho_summary(
    aggregate_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    rho_diagnostics: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for model_name in (
        "champion_dc_xg",
        "dc_rho_mild_minus_08",
        "dc_fit_rho_each_fold",
        "dc_fit_rho_each_fold_wide_grid",
    ):
        aggregate = find_row(aggregate_metrics, model_name)
        folds = fold_metrics[fold_metrics["model_name"] == model_name]
        rho_rows = rho_diagnostics[rho_diagnostics["model_name"] == model_name]
        fitted = rho_rows["resolved_rho"].dropna().astype(float)
        boundary_rows = rho_rows[rho_rows["rho_hits_grid_min"] | rho_rows["rho_hits_grid_max"]]
        rows.append(
            round_record(
                {
                    **aggregate,
                    "rho_min": float(fitted.min()) if not fitted.empty else None,
                    "rho_max": float(fitted.max()) if not fitted.empty else None,
                    "rho_mean": float(fitted.mean()) if not fitted.empty else None,
                    "rho_median": float(fitted.median()) if not fitted.empty else None,
                    "rho_std": float(fitted.std(ddof=0)) if not fitted.empty else None,
                    "rho_grid_min": single_value(rho_rows, "rho_grid_min"),
                    "rho_grid_max": single_value(rho_rows, "rho_grid_max"),
                    "grid_min_hits": int(rho_rows["rho_hits_grid_min"].sum()),
                    "grid_max_hits": int(rho_rows["rho_hits_grid_max"].sum()),
                    "boundary_hit_fold_ids": boundary_rows["fold_id"].tolist(),
                    "boundary_hit_train_sizes": boundary_rows["train_size"].astype(int).tolist(),
                    "folds_better_log_loss_vs_champion": int((folds["log_loss_delta_vs_champion"] < 0).sum()),
                    "folds_worse_log_loss_vs_champion": int((folds["log_loss_delta_vs_champion"] > 0).sum()),
                    "folds_better_brier_vs_champion": int((folds["brier_delta_vs_champion"] < 0).sum()),
                    "folds_worse_brier_vs_champion": int((folds["brier_delta_vs_champion"] > 0).sum()),
                    "mean_log_loss_delta_vs_champion": float(folds["log_loss_delta_vs_champion"].mean()),
                    "mean_brier_delta_vs_champion": float(folds["brier_delta_vs_champion"].mean()),
                }
            )
        )
    return rows


def build_combined_summary(
    aggregate_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    combined_spec: DixonColesVariantSpec | None,
) -> dict[str, Any] | None:
    if combined_spec is None:
        return None
    aggregate = find_row(aggregate_metrics, combined_spec.model_name)
    folds = fold_metrics[fold_metrics["model_name"] == combined_spec.model_name]
    return round_record(
        {
            **aggregate,
            "folds_better_log_loss_vs_champion": int((folds["log_loss_delta_vs_champion"] < 0).sum()),
            "folds_worse_log_loss_vs_champion": int((folds["log_loss_delta_vs_champion"] > 0).sum()),
            "folds_better_brier_vs_champion": int((folds["brier_delta_vs_champion"] < 0).sum()),
            "folds_worse_brier_vs_champion": int((folds["brier_delta_vs_champion"] > 0).sum()),
            "mean_log_loss_delta_vs_champion": float(folds["log_loss_delta_vs_champion"].mean()),
            "mean_brier_delta_vs_champion": float(folds["brier_delta_vs_champion"].mean()),
        }
    )


def build_observations(
    *,
    aggregate_metrics: pd.DataFrame,
    pseudocount_summary: list[dict[str, Any]],
    rho_summary: list[dict[str, Any]],
    combined_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    champion = find_row(aggregate_metrics, "champion_dc_xg")
    best_log = aggregate_metrics.sort_values(["log_loss", "brier_score"]).iloc[0].to_dict()
    best_brier = aggregate_metrics.sort_values(["brier_score", "log_loss"]).iloc[0].to_dict()
    beat_champion = aggregate_metrics[
        (aggregate_metrics["log_loss"] < champion["log_loss"])
        | (aggregate_metrics["brier_score"] < champion["brier_score"])
    ]
    return {
        "champion": round_record(champion),
        "best_by_log_loss": round_record(best_log),
        "best_by_brier": round_record(best_brier),
        "candidates_beating_champion": [round_record(row) for row in beat_champion.to_dict(orient="records")],
        "pseudocount_conclusion": pseudocount_conclusion(pseudocount_summary),
        "rho_conclusion": rho_conclusion(rho_summary),
        "combined_conclusion": combined_conclusion(combined_summary),
    }


def confidence_diagnostics(rows: pd.DataFrame) -> dict[str, Any]:
    probabilities = rows.loc[:, PROBABILITY_COLUMNS].astype(float).to_numpy()
    validate_probabilities(probabilities)
    outcomes = outcome_indices(rows["actual_outcome"].tolist())
    actual_probs = probabilities[np.arange(len(rows)), outcomes]
    max_probs = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-15, 1.0))).sum(axis=1)
    return {
        "average_max_probability": float(max_probs.mean()),
        "average_entropy": float(entropy.mean()),
        "high_confidence_wrong_count": int(((max_probs >= 0.65) & (predictions != outcomes)).sum()),
        "actual_probability_below_05_count": int((actual_probs < 0.05).sum()),
        "actual_probability_below_10_count": int((actual_probs < 0.10).sum()),
    }


def draw_diagnostics(rows: pd.DataFrame) -> dict[str, Any]:
    probabilities = rows.loc[:, PROBABILITY_COLUMNS].astype(float).to_numpy()
    outcomes = outcome_indices(rows["actual_outcome"].tolist())
    draws = outcomes == 1
    non_draws = ~draws
    predicted = probabilities.argmax(axis=1)
    return {
        "actual_draws": int(draws.sum()),
        "draw_prediction_rate": float((predicted == 1).mean()),
        "avg_predicted_draw_probability": float(probabilities[:, 1].mean()),
        "draw_recall": float(((predicted == 1) & draws).sum() / draws.sum()) if draws.sum() else None,
        "draw_log_loss": mean_log_loss(probabilities[draws], outcomes[draws]) if draws.sum() else None,
        "non_draw_log_loss": mean_log_loss(probabilities[non_draws], outcomes[non_draws]) if non_draws.sum() else None,
        "non_draw_accuracy": float((predicted[non_draws] == outcomes[non_draws]).mean()) if non_draws.sum() else None,
    }


def favourite_diagnostics(rows: pd.DataFrame) -> dict[str, Any]:
    probabilities = rows.loc[:, PROBABILITY_COLUMNS].astype(float).to_numpy()
    outcomes = outcome_indices(rows["actual_outcome"].tolist())
    predicted = probabilities.argmax(axis=1)
    favourites = np.where(predicted == 0, "home", np.where(predicted == 2, "away", "draw"))
    return {
        "home_favourite_count": int((favourites == "home").sum()),
        "away_favourite_count": int((favourites == "away").sum()),
        "draw_favourite_count": int((favourites == "draw").sum()),
        "favourite_accuracy": float((predicted == outcomes).mean()),
    }


def favourite_breakdown(rows: pd.DataFrame) -> list[dict[str, Any]]:
    probabilities = rows.loc[:, PROBABILITY_COLUMNS].astype(float).to_numpy()
    outcomes = outcome_indices(rows["actual_outcome"].tolist())
    predicted = probabilities.argmax(axis=1)
    favourite_labels = np.where(predicted == 0, "home", np.where(predicted == 2, "away", "draw"))
    result = []
    for label in ("home", "draw", "away"):
        mask = favourite_labels == label
        result.append(
            {
                "favourite": label,
                "count": int(mask.sum()),
                "accuracy": float((predicted[mask] == outcomes[mask]).mean()) if mask.sum() else None,
                "avg_max_probability": float(probabilities[mask].max(axis=1).mean()) if mask.sum() else None,
            }
        )
    return result


def mean_log_loss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    chosen = probabilities[np.arange(len(outcomes)), outcomes]
    return float(-np.log(np.clip(chosen, 1e-15, 1.0)).mean())


def render_markdown_report(payload: dict[str, Any]) -> str:
    observations = payload["observations"]
    lines = [
        "# Pseudocount and Rho Stability Diagnostics",
        "",
        "## Executive Summary",
        "",
        summary_sentence(observations),
        "",
        "No candidate should be promoted from this branch alone.",
        "",
        "## Inputs and Protocol",
        "",
        f"- CSV used: `{payload['csv_path']}`",
        f"- Evaluation window: {payload['backtest_config']['test_start']} to {payload['backtest_config']['test_end']}",
        f"- Minimum training matches: {payload['backtest_config']['min_train_matches']}",
        f"- Rolling folds: {len(payload['folds'])}",
        "- Train rows are strictly before each test window.",
        "- xG columns required and verified: `home_np_xg`, `away_np_xg`, `home_xg`, `away_xg`.",
        "",
        "## Aggregate Leaderboard",
        "",
        markdown_table(
            payload["aggregate_metrics"],
            [
                "rank",
                "model_name",
                "candidate_category",
                "log_loss",
                "brier_score",
                "accuracy",
                "average_max_probability",
                "average_entropy",
                "high_confidence_wrong_count",
                "actual_probability_below_05_count",
                "actual_probability_below_10_count",
            ],
        ),
        "",
        "## Pseudocount Stability",
        "",
        markdown_table(
            payload["pseudocount_summary"],
            [
                "xg_pseudocount",
                "model_name",
                "log_loss",
                "brier_score",
                "accuracy",
                "average_max_probability",
                "average_entropy",
                "folds_better_log_loss_vs_005",
                "folds_worse_log_loss_vs_005",
                "folds_better_brier_vs_005",
                "folds_worse_brier_vs_005",
                "draw_recall",
                "draw_log_loss",
            ],
        ),
        "",
        observations["pseudocount_conclusion"],
        "",
        "## Fitted-Rho Stability",
        "",
        markdown_table(
            payload["rho_summary"],
            [
                "model_name",
                "log_loss",
                "brier_score",
                "accuracy",
                "rho_min",
                "rho_max",
                "rho_mean",
                "rho_median",
                "rho_std",
                "grid_min_hits",
                "grid_max_hits",
                "folds_better_log_loss_vs_champion",
                "folds_worse_log_loss_vs_champion",
            ],
        ),
        "",
        observations["rho_conclusion"],
        "",
        "## Combined Diagnostic",
        "",
        combined_section(payload["combined_summary"], observations["combined_conclusion"]),
        "",
        "## Candidates Beating Champion",
        "",
        markdown_table(
            observations["candidates_beating_champion"],
            ["model_name", "candidate_category", "log_loss", "brier_score", "accuracy"],
        ),
        "",
        "## Carry-Forward Recommendation",
        "",
        recommendation_section(payload),
        "",
        "## Guardrails",
        "",
        "- The operational champion implementation remains frozen.",
        "- The wider rho grid row is diagnostic only.",
        "- The `0.25` pseudocount row is a saturation check, not a broad search.",
        "- Market odds, neural networks, embeddings, and broad grids remain out of scope.",
        "",
    ]
    return "\n".join(lines)


def summary_sentence(observations: dict[str, Any]) -> str:
    best_log = observations["best_by_log_loss"]
    best_brier = observations["best_by_brier"]
    champion = observations["champion"]
    return (
        f"Best by log loss: `{best_log['model_name']}` ({best_log['log_loss']:.4f}). "
        f"Best by Brier: `{best_brier['model_name']}` ({best_brier['brier_score']:.4f}). "
        f"Champion reference: log loss {champion['log_loss']:.4f}, Brier {champion['brier_score']:.4f}."
    )


def pseudocount_conclusion(rows: list[dict[str, Any]]) -> str:
    best = min(rows, key=lambda row: (row["log_loss"], row["brier_score"]))
    champion = next(row for row in rows if row["xg_pseudocount"] == 0.05)
    row_025 = next(row for row in rows if row["xg_pseudocount"] == 0.25)
    row_020 = next(row for row in rows if row["xg_pseudocount"] == 0.20)
    if row_025["log_loss"] < row_020["log_loss"] and row_025["brier_score"] <= row_020["brier_score"]:
        saturation = "`0.25` continued the probability-quality improvement, so the saturation boundary did not reverse."
    else:
        saturation = "`0.25` flattened or reversed versus `0.20`, so `0.20` remains the cleaner carry-forward point."
    draw_warning = ""
    if best["draw_recall"] < champion["draw_recall"]:
        draw_warning = (
            f" However, `{best['model_name']}` reduced draw recall from {champion['draw_recall']:.4f} "
            f"to {best['draw_recall']:.4f}, so it is not the cleanest shadow candidate."
        )
    return (
        f"Best pseudocount row: `{best['model_name']}`. "
        f"It beat the 0.05 reference on {best['folds_better_log_loss_vs_005']} folds by log loss and "
        f"{best['folds_better_brier_vs_005']} folds by Brier. {saturation}{draw_warning}"
    )


def rho_conclusion(rows: list[dict[str, Any]]) -> str:
    current = next(row for row in rows if row["model_name"] == "dc_fit_rho_each_fold")
    wide = next(row for row in rows if row["model_name"] == "dc_fit_rho_each_fold_wide_grid")
    return (
        f"Current fitted rho improved on {current['folds_better_log_loss_vs_champion']} folds by log loss, "
        f"with {current['grid_max_hits']} max-boundary hits. The wide-grid diagnostic had "
        f"{wide['grid_max_hits']} max-boundary hits and rho max {wide['rho_max']:.4f}. "
        "Treat fitted rho as diagnostic unless boundary behavior is stable under future validation."
    )


def combined_conclusion(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "No combined candidate was run because neither `0.20` nor `0.25` passed the stability gate."
    return (
        f"`{summary['model_name']}` was run as diagnostic only. It beat champion on "
        f"{summary['folds_better_log_loss_vs_champion']} folds by log loss and "
        f"{summary['folds_better_brier_vs_champion']} folds by Brier."
    )


def combined_section(summary: dict[str, Any] | None, conclusion: str) -> str:
    if summary is None:
        return conclusion
    table = markdown_table(
        [summary],
        [
            "model_name",
            "log_loss",
            "brier_score",
            "accuracy",
            "average_max_probability",
            "average_entropy",
            "folds_better_log_loss_vs_champion",
            "folds_better_brier_vs_champion",
        ],
    )
    return f"{table}\n\n{conclusion}"


def recommendation_section(payload: dict[str, Any]) -> str:
    champion = next(row for row in payload["pseudocount_summary"] if row["xg_pseudocount"] == 0.05)
    safe_pseudo_rows = [
        row
        for row in payload["pseudocount_summary"]
        if row["xg_pseudocount"] > 0.05
        and row["log_loss"] < champion["log_loss"]
        and row["brier_score"] < champion["brier_score"]
        and row["draw_recall"] >= champion["draw_recall"]
    ]
    pseudo_best = min(safe_pseudo_rows, key=lambda row: (row["log_loss"], row["brier_score"]))
    probability_best = min(payload["pseudocount_summary"], key=lambda row: (row["log_loss"], row["brier_score"]))
    lines = [
        f"- Carry forward `{pseudo_best['model_name']}` as the safest pseudocount shadow hypothesis.",
        f"- Keep `{probability_best['model_name']}` as a probability-quality diagnostic because draw recall fell to zero.",
        "- Keep fitted-rho rows diagnostic until boundary behavior is better understood.",
    ]
    if payload["combined_summary"] is not None:
        lines.append(f"- Keep `{payload['combined_summary']['model_name']}` diagnostic-only; it stacks two uncertain levers.")
    lines.append("- Park decay changes for this branch; they were not re-tested here.")
    return "\n".join(lines)


def write_outputs(payload: dict[str, Any], *, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{prefix}.md").write_text(payload["markdown"], encoding="utf-8")
    json_payload = {key: value for key, value in payload.items() if key != "markdown"}
    (output_dir / f"{prefix}.json").write_text(
        json.dumps(json_payload, default=json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(payload["fold_metrics"]).to_csv(output_dir / "pseudocount_rho_fold_metrics.csv", index=False)
    pd.DataFrame(payload["rho_diagnostics"]).to_csv(output_dir / "pseudocount_rho_rho_by_fold.csv", index=False)


def candidate_payload(spec_def: DixonColesVariantSpec) -> dict[str, Any]:
    config = asdict(build_model_config(spec_def))
    config["rho_grid"] = list(config["rho_grid"])
    return {
        "model_name": spec_def.model_name,
        "candidate_category": candidate_category(spec_def.model_name),
        "description": spec_def.description,
        "config_overrides": dict(spec_def.config_overrides),
        "rho_behavior": spec_def.rho_behavior,
        "config": config,
        "offline_only": True,
    }


def candidate_category(model_name: str) -> str:
    if model_name == "champion_dc_xg":
        return "champion_reference"
    if model_name.startswith("txg_xg_pseudocount"):
        return "pseudocount_stability"
    if model_name == "dc_fit_rho_xg_pseudocount_020" or model_name == "dc_fit_rho_xg_pseudocount_025":
        return "combined_diagnostic"
    if model_name == "dc_fit_rho_each_fold_wide_grid":
        return "rho_boundary_diagnostic"
    return "rho_stability"


def resolved_model_rho(model: DixonColesVariantModel) -> float | None:
    champion = getattr(model, "_champion", None)
    rho = getattr(champion, "rho", None)
    return float(rho) if rho is not None else None


def find_row(df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    rows = df[df["model_name"] == model_name]
    if rows.empty:
        raise ValueError(f"Missing model row: {model_name}")
    return rows.iloc[0].to_dict()


def single_value(rows: pd.DataFrame, column: str) -> Any:
    values = rows[column].dropna().unique().tolist()
    return values[0] if values else None


def round_record(row: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in row.items():
        if isinstance(value, float):
            result[key] = round(value, 6)
        else:
            result[key] = value
    return result


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    if not rows:
        return "\n".join([header, divider])
    body = ["| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


if __name__ == "__main__":
    main()
