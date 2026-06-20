#!/usr/bin/env python3
"""
Run a local rolling backtest experiment scaffold.

This script intentionally reads from a local CSV and does not require live
Supabase access. It currently supports the frozen champion adapter; future
challengers can be added by extending ``_model_provider``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.backtesting import BacktestConfig, build_rolling_folds, config_to_dict, run_backtest_for_model
from models.champion_dc_xg import ChampionDCXGModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local rolling backtest experiment.")
    parser.add_argument("--csv", required=True, help="Local match-data CSV path.")
    parser.add_argument("--model", default="champion_dc_xg", choices=["champion_dc_xg"])
    parser.add_argument("--test-start", required=True, help="First test-window date, YYYY-MM-DD.")
    parser.add_argument("--test-end", required=True, help="Final test-window date, YYYY-MM-DD.")
    parser.add_argument("--train-start", default=None, help="Optional earliest training date, YYYY-MM-DD.")
    parser.add_argument("--test-window-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--min-train-matches", type=int, default=10)
    parser.add_argument("--round-label", action="append", dest="round_labels", help="Optional test round label filter.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--dry-run", action="store_true", help="Build folds and metadata without fitting a model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df["match_date"] = pd.to_datetime(df["match_date"], format="ISO8601", errors="raise")
    config = BacktestConfig(
        test_start=args.test_start,
        test_end=args.test_end,
        train_start=args.train_start,
        test_window_days=args.test_window_days,
        step_days=args.step_days,
        min_train_matches=args.min_train_matches,
        round_labels=tuple(args.round_labels) if args.round_labels else None,
    )
    folds = build_rolling_folds(df, config)

    if args.dry_run:
        payload = {
            "model": args.model,
            "backtest_config": config_to_dict(config),
            "folds": [fold.metadata() for fold in folds],
        }
    else:
        result = run_backtest_for_model(_model_provider(args.model), df, folds)
        payload = {
            "backtest_config": config_to_dict(config),
            **result.to_dict(),
        }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


def _model_provider(model_name: str):
    if model_name == "champion_dc_xg":
        return ChampionDCXGModel
    raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    main()
