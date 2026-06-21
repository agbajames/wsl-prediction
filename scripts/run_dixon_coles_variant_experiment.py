#!/usr/bin/env python3
"""
Run offline Dixon-Coles/champion-family configuration variant experiments.

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

from evaluation.backtesting import BacktestConfig
from evaluation.dixon_coles_variants import available_variant_names, run_dixon_coles_variant_experiment
from scripts.run_model_comparison import load_match_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Dixon-Coles variant experiment.")
    parser.add_argument("--csv", required=True, help="Local match-data CSV path.")
    parser.add_argument("--variant", action="append", dest="variants", help="Variant name to include. Repeatable.")
    parser.add_argument("--test-start", required=True, help="First test-window date, YYYY-MM-DD.")
    parser.add_argument("--test-end", required=True, help="Final test-window date, YYYY-MM-DD.")
    parser.add_argument("--train-start", default=None, help="Optional earliest training date, YYYY-MM-DD.")
    parser.add_argument("--test-window-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--min-train-matches", type=int, default=10)
    parser.add_argument("--output-md", required=True, help="Markdown report output path.")
    parser.add_argument("--output-json", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--list-variants", action="store_true", help="Print available variants and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_variants:
        print("\n".join(available_variant_names()))
        return

    df = load_match_csv(Path(args.csv))
    config = BacktestConfig(
        test_start=args.test_start,
        test_end=args.test_end,
        train_start=args.train_start,
        test_window_days=args.test_window_days,
        step_days=args.step_days,
        min_train_matches=args.min_train_matches,
    )
    payload = run_dixon_coles_variant_experiment(
        df,
        backtest_config=config,
        variant_names=tuple(args.variants) if args.variants else None,
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


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


if __name__ == "__main__":
    main()
