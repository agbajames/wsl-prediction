#!/usr/bin/env python3
"""
Run the evaluation-only market-implied benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.market_benchmark import build_market_benchmark_result, load_market_odds_csv, write_market_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a de-vigged market probability benchmark CSV.")
    parser.add_argument("--csv", required=True, help="Local market odds CSV path.")
    parser.add_argument("--include-non-league", action="store_true", help="Include rows with non-empty Note values.")
    parser.add_argument("--output-md", required=True, help="Markdown report output path.")
    parser.add_argument("--output-json", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--output-rows", default=None, help="Optional row-level CSV output path.")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_market_odds_csv(Path(args.csv))
    result = build_market_benchmark_result(
        raw,
        include_non_league=args.include_non_league,
        n_bins=args.n_bins,
        top_n=args.top_n,
    )
    write_market_outputs(
        result,
        output_md=args.output_md,
        output_json=args.output_json,
        output_rows=args.output_rows,
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
