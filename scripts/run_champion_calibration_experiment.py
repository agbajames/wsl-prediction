#!/usr/bin/env python3
"""
Run an offline champion probability-calibration experiment.

The script reads a local model-comparison JSON artefact and does not require
live Supabase access.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.calibrators import (
    build_calibration_experiment,
    load_model_comparison_payload,
    render_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run champion calibration experiment from local JSON.")
    parser.add_argument("--input-json", required=True, help="Model comparison JSON artefact.")
    parser.add_argument("--output-md", required=True, help="Markdown report output path.")
    parser.add_argument("--output-json", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--champion-model", default="champion_dc_xg", help="Champion model name to calibrate.")
    parser.add_argument("--calibration-fraction", type=float, default=0.6, help="Share of earlier folds used for fitting.")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_model_comparison_payload(Path(args.input_json))
    summary = build_calibration_experiment(
        payload,
        champion_model=args.champion_model,
        calibration_fraction=args.calibration_fraction,
        n_bins=args.n_bins,
        top_n=args.top_n,
    )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown_report(summary), encoding="utf-8")

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(summary, default=_json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
