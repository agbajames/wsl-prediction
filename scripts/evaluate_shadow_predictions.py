#!/usr/bin/env python3
"""Replay saved shadow predictions once fixture results are known."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.shadow import evaluate_shadow_predictions, load_fixture_file, load_shadow_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate shadow prediction artefacts against results.")
    parser.add_argument("--predictions", required=True, help="Saved shadow predictions .json or .csv path.")
    parser.add_argument("--results", required=True, help="Results CSV or JSON with fixture IDs and outcomes/goals.")
    parser.add_argument("--output-json", required=True, help="Replay/evaluation JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_shadow_predictions(Path(args.predictions))
    results = load_fixture_file(Path(args.results))
    payload = evaluate_shadow_predictions(predictions, results)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


if __name__ == "__main__":
    main()
