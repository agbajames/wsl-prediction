#!/usr/bin/env python3
"""
Run an offline matched-fixture model-vs-market comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.model_market_comparison import (
    build_model_market_comparison,
    load_model_prediction_rows,
    write_model_market_outputs,
)
from evaluation.market_benchmark import load_market_odds_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model predictions to a market-implied benchmark.")
    parser.add_argument("--model-json", required=True, help="Model comparison JSON with prediction_rows.")
    parser.add_argument("--market-csv", required=True, help="Local market odds CSV path.")
    parser.add_argument("--output-md", required=True, help="Markdown report output path.")
    parser.add_argument("--output-json", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--output-rows", default=None, help="Optional row-level CSV output path.")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_rows = load_model_prediction_rows(Path(args.model_json))
    market_rows = load_market_odds_csv(Path(args.market_csv))
    result = build_model_market_comparison(
        model_rows,
        market_rows,
        n_bins=args.n_bins,
        top_n=args.top_n,
    )
    write_model_market_outputs(
        result,
        output_md=args.output_md,
        output_json=args.output_json,
        output_rows=args.output_rows,
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
