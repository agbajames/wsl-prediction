#!/usr/bin/env python3
"""Generate timestamped shadow predictions for upcoming WSL fixtures."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.shadow import (
    DEFAULT_SHADOW_MODELS,
    current_git_sha,
    generate_shadow_predictions,
    load_fixture_file,
    write_shadow_predictions,
)
from scripts.run_model_comparison import load_match_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pre-match shadow prediction artefacts.")
    parser.add_argument("--history-csv", required=True, help="Historical match-data CSV path.")
    parser.add_argument("--fixtures", required=True, help="Upcoming fixtures CSV or JSON path.")
    parser.add_argument("--model", action="append", dest="models", help="Model name to track. Repeatable.")
    parser.add_argument("--min-train-matches", type=int, default=10)
    parser.add_argument("--prediction-timestamp", default=None, help="Optional UTC timestamp override for tests/audits.")
    parser.add_argument("--output", required=True, help="Output .json or .csv artefact path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    historical = load_match_csv(Path(args.history_csv))
    fixtures = load_fixture_file(Path(args.fixtures))
    model_names = tuple(args.models) if args.models else DEFAULT_SHADOW_MODELS
    sha = current_git_sha()
    predictions = generate_shadow_predictions(
        historical,
        fixtures,
        model_names=model_names,
        prediction_timestamp=args.prediction_timestamp,
        git_sha=sha,
        min_train_matches=args.min_train_matches,
    )
    write_shadow_predictions(
        predictions,
        Path(args.output),
        metadata={
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "git_sha": sha,
            "history_csv": str(args.history_csv),
            "fixtures": str(args.fixtures),
            "models": list(model_names),
            "min_train_matches": args.min_train_matches,
        },
    )

if __name__ == "__main__":
    main()
