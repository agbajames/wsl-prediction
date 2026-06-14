"""
evaluation/run_evaluation.py
----------------------------
Repeatable evaluation runner for WSL prediction backtests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

import pandas as pd

# Allow direct execution: python evaluation/run_evaluation.py ...
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.supabase_client import fetch_match_data, get_supabase_client
from evaluation.metrics import evaluate_prediction_set
from model.wsl_xg_model import ModelConfig, run_backtest


def _per_match_probabilities_and_outcomes(per_match: list[dict[str, Any]]) -> tuple[list[list[float]], list[str]]:
    probabilities: list[list[float]] = []
    outcomes: list[str] = []
    for row in per_match:
        probabilities.append([float(row["p_home"]), float(row["p_draw"]), float(row["p_away"])])
        outcomes.append(str(row["outcome"]))
    return probabilities, outcomes


def run_walk_forward_evaluation(
    df: pd.DataFrame | None = None,
    *,
    start_date: str | pd.Timestamp,
    alpha: float = 0.15,
    decay_days: float = 60.0,
    rho: float | None = -0.13,
    fit_rho_each_batch: bool = False,
    min_training_matches: int = 10,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Run walk-forward evaluation and return a structured JSON-style result.

    If `df` is not provided, match data is fetched through the existing
    Supabase data layer. Tests should pass a local DataFrame fixture.
    """
    if df is None:
        df = fetch_match_data(get_supabase_client())

    config = ModelConfig(alpha=alpha, decay_half_life_days=decay_days, rho=rho)
    start_ts = pd.Timestamp(start_date)
    backtest = run_backtest(
        df,
        config,
        start_date=start_ts,
        fit_rho_each_batch=fit_rho_each_batch,
        min_training_matches=min_training_matches,
    )

    probabilities, outcomes = _per_match_probabilities_and_outcomes(backtest.per_match)
    metrics = (
        evaluate_prediction_set(probabilities, outcomes, n_bins=n_bins)
        if probabilities
        else {
            "n_matches": 0,
            "brier_score": 0.0,
            "log_loss": 0.0,
            "accuracy": 0.0,
            "calibration_bins": [],
            "confidence_buckets": [],
        }
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "walk_forward",
        "parameters": {
            "start_date": start_ts.date().isoformat(),
            "alpha": alpha,
            "decay_days": decay_days,
            "rho": rho,
            "fit_rho_each_batch": fit_rho_each_batch,
            "min_training_matches": min_training_matches,
            "n_bins": n_bins,
        },
        "metrics": metrics,
        "model_backtest_metrics": {
            "n_matches": backtest.n_matches,
            "brier_score": backtest.brier_score,
            "log_loss": backtest.log_loss,
            "calibration_bins": backtest.calibration_bins,
        },
        "per_match_results": backtest.per_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WSL walk-forward model evaluation.")
    parser.add_argument("--start-date", required=True, help="First match date to evaluate (YYYY-MM-DD).")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--decay-days", type=float, default=60.0)
    parser.add_argument("--rho", type=float, default=-0.13)
    parser.add_argument("--fit-rho", action="store_true")
    parser.add_argument("--min-training-matches", type=int, default=10)
    parser.add_argument("--n-bins", type=int, default=5)
    args = parser.parse_args()

    result = run_walk_forward_evaluation(
        start_date=args.start_date,
        alpha=args.alpha,
        decay_days=args.decay_days,
        rho=None if args.fit_rho else args.rho,
        fit_rho_each_batch=args.fit_rho,
        min_training_matches=args.min_training_matches,
        n_bins=args.n_bins,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
