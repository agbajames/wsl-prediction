#!/usr/bin/env python3
"""
Run controlled neural-network challenger experiments on rolling time splits.

The script is intentionally small-grid and dependency-light. It evaluates a
predeclared architecture ladder for the research-only neural-network challenger
and writes diagnostics for probability quality, overfitting, and seed stability.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.backtesting import BacktestConfig, BacktestFold, build_rolling_folds, config_to_dict
from evaluation.metrics import (
    OUTCOME_LABELS,
    brier_score_3way,
    multiclass_log_loss,
    outcome_accuracy,
    outcome_indices,
    validate_probabilities,
)
from experiments.registry import get_model_constructor
from models.neural_network import NeuralNetworkChallenger

PROBABILITY_COLUMNS = ("p_home_win", "p_draw", "p_away_win")
XG_SOURCE_COLUMNS = ("home_np_xg", "away_np_xg", "home_xg", "away_xg")
DEFAULT_SEEDS = (42, 7, 123)
DEFAULT_BASELINES = (
    "champion_dc_xg",
    "improved_logistic_regression",
    "random_forest",
    "poisson_regression",
    "regularised_team_strength",
    "naive_outcome_rate",
)


@dataclass(frozen=True)
class NNExperimentConfig:
    name: str
    hidden_layers: tuple[int, ...]
    feature_group: str = "xg"


NN_EXPERIMENTS = (
    NNExperimentConfig("nn_logistic_xg", ()),
    NNExperimentConfig("nn_tiny_8_xg", (8,)),
    NNExperimentConfig("nn_tiny_16_xg", (16,)),
    NNExperimentConfig("nn_moderate_32_xg", (32,)),
    NNExperimentConfig("nn_moderate_64_xg", (64,)),
    NNExperimentConfig("nn_two_layer_32_16_xg", (32, 16)),
    NNExperimentConfig("nn_two_layer_64_32_xg", (64, 32)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled neural-network challenger experiments.")
    parser.add_argument("--csv", required=True, help="Local match-data CSV path.")
    parser.add_argument("--test-start", required=True, help="First test-window date, YYYY-MM-DD.")
    parser.add_argument("--test-end", required=True, help="Final test-window date, YYYY-MM-DD.")
    parser.add_argument("--train-start", default=None, help="Optional earliest training date, YYYY-MM-DD.")
    parser.add_argument("--test-window-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--min-train-matches", type=int, default=12)
    parser.add_argument("--round-label", action="append", dest="round_labels")
    parser.add_argument("--seed", action="append", dest="seeds", type=int, help="Seed to run. Repeatable.")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-penalty", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--baseline-model", action="append", dest="baseline_models")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument(
        "--allow-xg-fallback-to-goals",
        action="store_true",
        help=(
            "Allow xg feature experiments to run when xG columns are absent or incomplete. "
            "Use only for explicit fallback-goals diagnostics."
        ),
    )
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--prefix", default="neural_network_experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_match_csv(Path(args.csv))
    folds = build_rolling_folds(
        df,
        BacktestConfig(
            test_start=args.test_start,
            test_end=args.test_end,
            train_start=args.train_start,
            test_window_days=args.test_window_days,
            step_days=args.step_days,
            min_train_matches=args.min_train_matches,
            round_labels=tuple(args.round_labels) if args.round_labels else None,
        ),
    )
    if not folds:
        raise ValueError("No backtest folds were created. Check dates and min_train_matches.")

    payload = run_neural_network_experiments(
        df,
        folds=folds,
        backtest_config=BacktestConfig(
            test_start=args.test_start,
            test_end=args.test_end,
            train_start=args.train_start,
            test_window_days=args.test_window_days,
            step_days=args.step_days,
            min_train_matches=args.min_train_matches,
            round_labels=tuple(args.round_labels) if args.round_labels else None,
        ),
        seeds=tuple(args.seeds) if args.seeds else DEFAULT_SEEDS,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        l2_penalty=args.l2_penalty,
        dropout=args.dropout,
        validation_fraction=args.validation_fraction,
        early_stopping_patience=args.early_stopping_patience,
        baseline_models=() if args.skip_baselines else tuple(args.baseline_models or DEFAULT_BASELINES),
        allow_xg_fallback_to_goals=args.allow_xg_fallback_to_goals,
    )
    write_outputs(payload, output_dir=Path(args.output_dir), prefix=args.prefix)


def load_match_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["match_date"] = pd.to_datetime(df["match_date"], format="ISO8601", errors="raise")
    return df


def run_neural_network_experiments(
    df: pd.DataFrame,
    *,
    folds: list[BacktestFold],
    backtest_config: BacktestConfig,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    max_iter: int = 500,
    learning_rate: float = 0.03,
    l2_penalty: float = 1e-3,
    dropout: float = 0.0,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 20,
    baseline_models: tuple[str, ...] = DEFAULT_BASELINES,
    allow_xg_fallback_to_goals: bool = False,
) -> dict[str, Any]:
    _validate_feature_inputs(df, allow_xg_fallback_to_goals=allow_xg_fallback_to_goals)
    prediction_frames: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    seed_metric_rows: list[dict[str, Any]] = []
    training_history_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for config in NN_EXPERIMENTS:
        for seed in seeds:
            model_predictions, histories = _run_nn_config(
                df,
                folds=folds,
                config=config,
                seed=seed,
                max_iter=max_iter,
                learning_rate=learning_rate,
                l2_penalty=l2_penalty,
                dropout=dropout,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
            )
            if model_predictions.empty:
                continue
            prediction_frames.append(model_predictions)
            training_history_rows.extend(histories)
            seed_metric_rows.append(_metric_row(model_predictions, experiment_name=config.name, seed=seed))
            fold_metric_rows.extend(_fold_metric_rows(model_predictions, experiment_name=config.name, seed=seed))

    for model_name in baseline_models:
        try:
            baseline_predictions = _run_baseline_model(df, folds=folds, model_name=model_name)
        except Exception as exc:  # noqa: BLE001 - report unavailable baselines without stopping NN runs.
            warnings.append({"model_name": model_name, "warning": str(exc)})
            continue
        if baseline_predictions.empty:
            continue
        prediction_frames.append(baseline_predictions)
        seed_metric_rows.append(_metric_row(baseline_predictions, experiment_name=model_name, seed=None))
        fold_metric_rows.extend(_fold_metric_rows(baseline_predictions, experiment_name=model_name, seed=None))

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    if predictions.empty:
        raise ValueError("No experiment predictions were generated.")

    seed_metrics = pd.DataFrame(seed_metric_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    training_history = pd.DataFrame(training_history_rows)
    summary = _summary_rows(seed_metrics)
    markdown = render_markdown_report(summary, seed_metrics, warnings)
    return {
        "backtest_config": config_to_dict(backtest_config),
        "data_schema": _data_schema_summary(df, allow_xg_fallback_to_goals=allow_xg_fallback_to_goals),
        "folds": [fold.metadata() for fold in folds],
        "summary": summary,
        "seed_metrics": seed_metrics,
        "fold_metrics": fold_metrics,
        "predictions": predictions,
        "training_history": training_history,
        "warnings": warnings,
        "markdown": markdown,
    }


def _validate_feature_inputs(df: pd.DataFrame, *, allow_xg_fallback_to_goals: bool) -> None:
    uses_xg = any(config.feature_group == "xg" for config in NN_EXPERIMENTS)
    if not uses_xg or allow_xg_fallback_to_goals:
        return

    missing = [column for column in XG_SOURCE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "xg neural-network experiments require xG source columns by default. "
            f"Missing columns: {missing}. Use --allow-xg-fallback-to-goals only for explicit diagnostics."
        )

    completed = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()
    home_has_xg = completed[["home_np_xg", "home_xg"]].notna().any(axis=1)
    away_has_xg = completed[["away_np_xg", "away_xg"]].notna().any(axis=1)
    if not bool((home_has_xg & away_has_xg).all()):
        missing_count = int((~(home_has_xg & away_has_xg)).sum())
        raise ValueError(
            "xg neural-network experiments require an xG value for both teams on every completed row by default. "
            f"Rows that would fall back to goals: {missing_count}. "
            "Use --allow-xg-fallback-to-goals only for explicit diagnostics."
        )


def _data_schema_summary(df: pd.DataFrame, *, allow_xg_fallback_to_goals: bool) -> dict[str, Any]:
    dates = pd.to_datetime(df["match_date"], format="ISO8601", errors="coerce")
    present = [column for column in XG_SOURCE_COLUMNS if column in df.columns]
    completed = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()
    fallback_rows = None
    if set(XG_SOURCE_COLUMNS).issubset(df.columns):
        home_has_xg = completed[["home_np_xg", "home_xg"]].notna().any(axis=1)
        away_has_xg = completed[["away_np_xg", "away_xg"]].notna().any(axis=1)
        fallback_rows = int((~(home_has_xg & away_has_xg)).sum())
    return {
        "row_count": int(len(df)),
        "completed_row_count": int(len(completed)),
        "min_match_date": dates.min().date().isoformat() if pd.notna(dates.min()) else None,
        "max_match_date": dates.max().date().isoformat() if pd.notna(dates.max()) else None,
        "xg_columns_present": present,
        "allow_xg_fallback_to_goals": allow_xg_fallback_to_goals,
        "rows_that_would_fallback_to_goals": fallback_rows,
    }


def _run_nn_config(
    df: pd.DataFrame,
    *,
    folds: list[BacktestFold],
    config: NNExperimentConfig,
    seed: int,
    max_iter: int,
    learning_rate: float,
    l2_penalty: float,
    dropout: float,
    validation_fraction: float,
    early_stopping_patience: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    prediction_frames: list[pd.DataFrame] = []
    training_history_rows: list[dict[str, Any]] = []
    for fold in folds:
        train = df.loc[list(fold.train_indices)].copy()
        test = df.loc[list(fold.test_indices)].copy()
        model = NeuralNetworkChallenger(
            hidden_layers=config.hidden_layers,
            learning_rate=learning_rate,
            l2_penalty=l2_penalty,
            dropout=dropout,
            max_iter=max_iter,
            min_training_matches=1,
            random_seed=seed,
            feature_group=config.feature_group,
            early_stopping=True,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
        ).fit(train)
        predictions = _attach_actual_results(model.predict(test), test)
        predictions["experiment_name"] = config.name
        predictions["seed"] = seed
        predictions["fold_id"] = fold.fold_id
        predictions["test_start"] = fold.test_start.date().isoformat()
        predictions["test_end"] = fold.test_end.date().isoformat()
        predictions["hidden_layers"] = _architecture_label(config.hidden_layers)
        predictions["dropout"] = dropout
        predictions["l2_penalty"] = l2_penalty
        prediction_frames.append(predictions)
        for row in model.training_history:
            training_history_rows.append(
                {
                    "experiment_name": config.name,
                    "seed": seed,
                    "fold_id": fold.fold_id,
                    "hidden_layers": _architecture_label(config.hidden_layers),
                    **row,
                    **model.fit_diagnostics,
                }
            )
    all_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return all_predictions, training_history_rows


def _run_baseline_model(df: pd.DataFrame, *, folds: list[BacktestFold], model_name: str) -> pd.DataFrame:
    constructor = get_model_constructor(model_name)
    prediction_frames = []
    for fold in folds:
        train = df.loc[list(fold.train_indices)].copy()
        test = df.loc[list(fold.test_indices)].copy()
        model = constructor().fit(train)
        predictions = _attach_actual_results(model.predict(test), test)
        predictions["experiment_name"] = model_name
        predictions["seed"] = np.nan
        predictions["fold_id"] = fold.fold_id
        predictions["test_start"] = fold.test_start.date().isoformat()
        predictions["test_end"] = fold.test_end.date().isoformat()
        predictions["hidden_layers"] = ""
        prediction_frames.append(predictions)
    return pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()


def _attach_actual_results(predictions: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    actual = test.reset_index(drop=True).copy()
    merged = predictions.reset_index(drop=True).copy()
    merged["actual_outcome"] = [
        _actual_outcome(row.home_goals, row.away_goals) for row in actual.itertuples(index=False)
    ]
    merged["home_goals"] = actual["home_goals"].to_numpy()
    merged["away_goals"] = actual["away_goals"].to_numpy()
    return merged


def _metric_row(predictions: pd.DataFrame, *, experiment_name: str, seed: int | None) -> dict[str, Any]:
    probs = predictions.loc[:, PROBABILITY_COLUMNS].astype(float).to_numpy()
    outcomes = predictions["actual_outcome"].tolist()
    diagnostics = _diagnostics(probs, outcomes)
    return {
        "experiment_name": experiment_name,
        "seed": seed,
        "n_matches": int(len(predictions)),
        "brier_score": brier_score_3way(probs, outcomes),
        "log_loss": multiclass_log_loss(probs, outcomes),
        "accuracy": outcome_accuracy(probs, outcomes),
        **diagnostics,
    }


def _fold_metric_rows(predictions: pd.DataFrame, *, experiment_name: str, seed: int | None) -> list[dict[str, Any]]:
    rows = []
    for fold_id, fold_predictions in predictions.groupby("fold_id", sort=True):
        rows.append({"fold_id": fold_id, **_metric_row(fold_predictions, experiment_name=experiment_name, seed=seed)})
    return rows


def _diagnostics(probabilities: np.ndarray, outcomes: list[str]) -> dict[str, Any]:
    probs = validate_probabilities(probabilities)
    actual = outcome_indices(outcomes)
    predicted = probs.argmax(axis=1)
    actual_probability = probs[np.arange(len(probs)), actual]
    confidence = probs.max(axis=1)
    confusion = _confusion_matrix(predicted, actual)
    per_class = _per_class_brier(probs, actual)
    return {
        "mean_p_home_win": float(probs[:, 0].mean()),
        "mean_p_draw": float(probs[:, 1].mean()),
        "mean_p_away_win": float(probs[:, 2].mean()),
        "average_max_confidence": float(confidence.mean()),
        "high_confidence_wrong_count": int(((confidence >= 0.8) & (predicted != actual)).sum()),
        "actual_probability_below_05_count": int((actual_probability < 0.05).sum()),
        "actual_probability_below_10_count": int((actual_probability < 0.10).sum()),
        **confusion,
        **per_class,
    }


def _confusion_matrix(predicted: np.ndarray, actual: np.ndarray) -> dict[str, int]:
    result = {}
    for actual_idx, actual_label in enumerate(OUTCOME_LABELS):
        for predicted_idx, predicted_label in enumerate(OUTCOME_LABELS):
            result[f"confusion_actual_{actual_label}_pred_{predicted_label}"] = int(
                ((actual == actual_idx) & (predicted == predicted_idx)).sum()
            )
    return result


def _per_class_brier(probabilities: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    result = {}
    for idx, label in enumerate(OUTCOME_LABELS):
        target = (actual == idx).astype(float)
        result[f"brier_{label}"] = float(np.mean((probabilities[:, idx] - target) ** 2))
    return result


def _summary_rows(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment_name, group in seed_metrics.groupby("experiment_name", sort=True):
        seed_rows = group.dropna(subset=["seed"])
        metric_group = seed_rows if not seed_rows.empty else group
        ordered = metric_group.sort_values(["log_loss", "brier_score", "accuracy"], ascending=[True, True, False])
        row = {
            "experiment_name": experiment_name,
            "n_runs": int(len(metric_group)),
            "n_matches": int(metric_group["n_matches"].max()),
            "mean_log_loss": float(metric_group["log_loss"].mean()),
            "std_log_loss": float(metric_group["log_loss"].std(ddof=0)) if len(metric_group) > 1 else 0.0,
            "mean_brier_score": float(metric_group["brier_score"].mean()),
            "std_brier_score": float(metric_group["brier_score"].std(ddof=0)) if len(metric_group) > 1 else 0.0,
            "mean_accuracy": float(metric_group["accuracy"].mean()),
            "std_accuracy": float(metric_group["accuracy"].std(ddof=0)) if len(metric_group) > 1 else 0.0,
            "best_seed": _seed_value(ordered.iloc[0].get("seed")),
            "worst_seed": _seed_value(ordered.iloc[-1].get("seed")),
            "seed_stable": bool(len(metric_group) <= 1 or metric_group["log_loss"].std(ddof=0) <= 0.02),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(["mean_log_loss", "mean_brier_score", "mean_accuracy"], ascending=[True, True, False])


def render_markdown_report(summary: pd.DataFrame, seed_metrics: pd.DataFrame, warnings: list[dict[str, Any]]) -> str:
    nn_summary = summary.loc[summary["experiment_name"].astype(str).str.startswith("nn_")]
    lines = [
        "# Neural Network Experiments",
        "",
        "This report evaluates a small predeclared NumPy neural-network architecture ladder as a challenger only. "
        "The champion model is not modified. Ranking prioritises log loss, then Brier score, then accuracy.",
        "",
        "## Summary",
        "",
        _markdown_table(
            summary.to_dict(orient="records"),
            [
                "experiment_name",
                "n_runs",
                "n_matches",
                "mean_log_loss",
                "std_log_loss",
                "mean_brier_score",
                "std_brier_score",
                "mean_accuracy",
                "std_accuracy",
                "best_seed",
                "worst_seed",
                "seed_stable",
            ],
        ),
        "",
        "## Neural Configurations Only",
        "",
        _markdown_table(
            nn_summary.to_dict(orient="records"),
            ["experiment_name", "mean_log_loss", "mean_brier_score", "mean_accuracy", "std_log_loss", "seed_stable"],
        ),
        "",
        "## Seed-Level Metrics",
        "",
        _markdown_table(
            seed_metrics.sort_values(["log_loss", "brier_score", "accuracy"], ascending=[True, True, False]).to_dict(
                orient="records"
            ),
            ["experiment_name", "seed", "n_matches", "log_loss", "brier_score", "accuracy"],
        ),
    ]
    if warnings:
        lines.extend(["", "## Warnings", "", _markdown_table(warnings, ["model_name", "warning"])])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], *, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload["summary"].to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    payload["fold_metrics"].to_csv(output_dir / f"{prefix}_fold_metrics.csv", index=False)
    payload["seed_metrics"].to_csv(output_dir / f"{prefix}_seed_metrics.csv", index=False)
    payload["predictions"].to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    payload["training_history"].to_csv(output_dir / f"{prefix}_training_history.csv", index=False)
    (output_dir / f"{prefix}.md").write_text(payload["markdown"], encoding="utf-8")
    json_payload = {
        "backtest_config": payload["backtest_config"],
        "data_schema": payload["data_schema"],
        "folds": payload["folds"],
        "warnings": payload["warnings"],
    }
    (output_dir / f"{prefix}_metadata.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def _actual_outcome(home_goals: Any, away_goals: Any) -> str:
    home = int(home_goals)
    away = int(away_goals)
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"


def _architecture_label(architecture: tuple[int, ...]) -> str:
    return "[]" if not architecture else "[" + ",".join(str(units) for units in architecture) + "]"


def _seed_value(value: Any) -> int | str:
    if pd.isna(value):
        return "baseline"
    return int(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not columns:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    if not rows:
        return "\n".join([header, divider])
    body = ["| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _format_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
