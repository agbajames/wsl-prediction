#!/usr/bin/env python3
"""
Run offline non-market fixed-weight blending experiments.

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
from evaluation.blending import (
    available_blend_names,
    build_default_component_providers,
    get_blend_spec,
    run_non_market_blending_experiment,
)
from scripts.run_model_comparison import load_match_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local non-market blending experiment.")
    parser.add_argument("--csv", required=True, help="Local match-data CSV path.")
    parser.add_argument("--blend", action="append", dest="blends", help="Blend name to include. Repeatable.")
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
    parser.add_argument("--list-blends", action="store_true", help="Print available blends and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_blends:
        print("\n".join(available_blend_names()))
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
    selected_specs = tuple(get_blend_spec(name) for name in args.blends) if args.blends else None
    selected_providers = None
    if selected_specs is not None:
        required_components = {component for spec in selected_specs for component in spec.components}
        default_providers = build_default_component_providers()
        selected_providers = {name: default_providers[name] for name in default_providers if name in required_components}

    payload = run_non_market_blending_experiment(
        df,
        backtest_config=config,
        component_providers=selected_providers,
        blend_specs=selected_specs,
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
