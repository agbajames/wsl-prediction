#!/usr/bin/env python3
"""
Generate a local evaluation report from prediction-vs-result rows.

The input can be CSV or JSON records and must include p_home_win, p_draw,
p_away_win, plus either an actual outcome column or home_goals/away_goals.
No live Supabase access is required.
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

from evaluation.calibration import calibration_summary
from evaluation.compare import compare_model_results, comparison_to_records, extract_outcomes, extract_probabilities
from evaluation.failure_analysis import failure_analysis_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local WSL evaluation report.")
    parser.add_argument("--input", required=True, help="CSV or JSON prediction-vs-result rows.")
    parser.add_argument("--output-md", required=True, help="Markdown report output path.")
    parser.add_argument("--output-json", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    summary = build_report_summary(rows, n_bins=args.n_bins, top_n=args.top_n)
    Path(args.output_md).write_text(render_markdown_report(summary), encoding="utf-8")
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(summary, default=_json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def load_rows(path: Path) -> pd.DataFrame:
    """Load local CSV or JSON rows."""
    if path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "predictions" in loaded:
            loaded = loaded["predictions"]
        return pd.DataFrame(loaded)
    return pd.read_csv(path)


def build_report_summary(rows: pd.DataFrame, *, n_bins: int = 5, top_n: int = 5) -> dict[str, Any]:
    """Build all report sections from local prediction-result rows."""
    probabilities = extract_probabilities(rows)
    outcomes = extract_outcomes(rows)
    comparison = compare_model_results(rows)
    return {
        "comparison": comparison_to_records(comparison),
        "calibration": calibration_summary(probabilities, outcomes, n_bins=n_bins, min_bin_size=2),
        "failure_analysis": failure_analysis_summary(rows, n=top_n),
    }


def render_markdown_report(summary: dict[str, Any]) -> str:
    """Render a concise Markdown evaluation report."""
    lines = [
        "# WSL Model Evaluation Report",
        "",
        "## Model Comparison",
        "",
        _markdown_table(
            summary["comparison"],
            ["rank", "model_name", "n_matches", "brier_score", "log_loss", "accuracy"],
        ),
        "",
        "## Calibration",
        "",
        _markdown_table(
            summary["calibration"]["calibration_bins"],
            ["bin", "count", "mean_confidence", "observed_accuracy", "calibration_gap", "is_sparse"],
        ),
        "",
        "## Confidence Buckets",
        "",
        _markdown_table(
            summary["calibration"]["confidence_buckets"],
            ["bucket", "count", "mean_confidence", "accuracy", "calibration_gap", "is_sparse"],
        ),
        "",
        "## Worst Misses",
        "",
        _markdown_table(
            summary["failure_analysis"]["worst_misses"],
            _failure_columns(summary["failure_analysis"]["worst_misses"]),
        ),
        "",
        "## Best High-Confidence Correct",
        "",
        _markdown_table(
            summary["failure_analysis"]["best_high_confidence_correct"],
            _failure_columns(summary["failure_analysis"]["best_high_confidence_correct"]),
        ),
        "",
        "## Favourite Breakdown",
        "",
        _markdown_table(
            summary["failure_analysis"]["favourite_breakdown"],
            [
                "predicted_outcome",
                "confidence_bucket",
                "n",
                "accuracy",
                "mean_confidence",
                "mean_actual_probability",
                "mean_log_loss",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _failure_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "model_name",
        "home_team",
        "away_team",
        "actual_outcome",
        "predicted_outcome",
        "predicted_confidence",
        "actual_probability",
        "row_log_loss",
    ]
    if not rows:
        return preferred
    return [column for column in preferred if column in rows[0]]


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not columns:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    if not rows:
        return "\n".join([header, divider])
    body = ["| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
