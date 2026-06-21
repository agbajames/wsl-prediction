#!/usr/bin/env python3
"""
Run local champion-vs-challenger model comparison on identical rolling folds.

The script reads a local CSV and does not require live Supabase access.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.backtesting import BacktestConfig, BacktestFold, build_rolling_folds, config_to_dict, run_backtest_for_model
from experiments.registry import available_models, get_model_constructor
from scripts.run_evaluation_report import build_report_summary, render_markdown_report

DEFAULT_MODELS = (
    "champion_dc_xg",
    "naive_outcome_rate",
    "elo_baseline",
    "logistic_regression",
    "regularised_team_strength",
    "poisson_regression",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local model comparison experiment.")
    parser.add_argument("--csv", required=True, help="Local match-data CSV path.")
    parser.add_argument("--model", action="append", dest="models", help="Model name to include. Repeatable.")
    parser.add_argument("--test-start", required=True, help="First test-window date, YYYY-MM-DD.")
    parser.add_argument("--test-end", required=True, help="Final test-window date, YYYY-MM-DD.")
    parser.add_argument("--train-start", default=None, help="Optional earliest training date, YYYY-MM-DD.")
    parser.add_argument("--test-window-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--min-train-matches", type=int, default=10)
    parser.add_argument("--round-label", action="append", dest="round_labels", help="Optional test round label filter.")
    parser.add_argument("--output-md", required=True, help="Markdown report output path.")
    parser.add_argument("--output-json", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--list-models", action="store_true", help="Print available models and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_models:
        print("\n".join(available_models()))
        return

    df = load_match_csv(Path(args.csv))
    model_names = tuple(args.models) if args.models else DEFAULT_MODELS
    config = BacktestConfig(
        test_start=args.test_start,
        test_end=args.test_end,
        train_start=args.train_start,
        test_window_days=args.test_window_days,
        step_days=args.step_days,
        min_train_matches=args.min_train_matches,
        round_labels=tuple(args.round_labels) if args.round_labels else None,
    )
    payload = run_comparison(
        df,
        model_names=model_names,
        backtest_config=config,
        n_bins=args.n_bins,
        top_n=args.top_n,
    )
    Path(args.output_md).write_text(payload["markdown"], encoding="utf-8")
    if args.output_json:
        json_payload = {key: value for key, value in payload.items() if key != "markdown"}
        Path(args.output_json).write_text(
            json.dumps(json_payload, default=_json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def load_match_csv(path: Path) -> pd.DataFrame:
    """Load a local match CSV with date parsing."""
    df = pd.read_csv(path)
    df["match_date"] = pd.to_datetime(df["match_date"], format="ISO8601", errors="raise")
    return df


def run_comparison(
    df: pd.DataFrame,
    *,
    model_names: tuple[str, ...],
    backtest_config: BacktestConfig,
    n_bins: int = 5,
    top_n: int = 5,
) -> dict[str, Any]:
    """Run requested models on shared folds and return report artefacts."""
    folds = build_rolling_folds(df, backtest_config)
    if not folds:
        raise ValueError("No backtest folds were created. Check dates and min_train_matches.")

    result_payloads = []
    prediction_frames = []
    for model_name in model_names:
        constructor = get_model_constructor(model_name)
        result = run_backtest_for_model(constructor, df, folds)
        predictions = attach_actual_results(result.predictions, df, folds)
        if not predictions.empty:
            prediction_frames.append(predictions)
        result_dict = result.to_dict()
        result_dict["predictions"] = predictions.to_dict(orient="records")
        result_payloads.append(result_dict)

    if not prediction_frames:
        raise ValueError("Models produced no predictions for the generated folds.")

    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    report_summary = build_report_summary(combined_predictions, n_bins=n_bins, top_n=top_n)
    markdown = render_markdown_report(report_summary)
    return {
        "backtest_config": config_to_dict(backtest_config),
        "folds": [fold.metadata() for fold in folds],
        "models": list(model_names),
        "model_results": result_payloads,
        "prediction_rows": combined_predictions.to_dict(orient="records"),
        "report_summary": report_summary,
        "markdown": markdown,
    }


def attach_actual_results(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    folds: list[BacktestFold],
) -> pd.DataFrame:
    """Attach actual outcomes to prediction rows in fold/test-row order."""
    if predictions.empty:
        return predictions

    actual_rows = []
    for fold in folds:
        actual = df.loc[list(fold.test_indices)].copy()
        for row in actual.itertuples(index=False):
            home_goals = getattr(row, "home_goals", None)
            away_goals = getattr(row, "away_goals", None)
            actual_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "actual_outcome": _actual_outcome(home_goals, away_goals),
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                }
            )

    actual_df = pd.DataFrame(actual_rows)
    if len(actual_df) != len(predictions):
        raise ValueError(
            "Prediction rows do not align with fold test rows: "
            f"{len(predictions)} predictions vs {len(actual_df)} actual rows."
        )
    merged = predictions.reset_index(drop=True).copy()
    for column in actual_df.columns:
        if column == "fold_id" and column in merged.columns:
            continue
        merged[column] = actual_df[column]
    return merged


def _actual_outcome(home_goals: Any, away_goals: Any) -> str:
    if pd.isna(home_goals) or pd.isna(away_goals):
        raise ValueError("Comparison rows require home_goals and away_goals for test fixtures.")
    home = int(home_goals)
    away = int(away_goals)
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


if __name__ == "__main__":
    main()

