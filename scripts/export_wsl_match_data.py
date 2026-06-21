#!/usr/bin/env python3
"""
Export WSL match-level data from Supabase to a local CSV.

This script uses the existing data access layer and prints only a safe summary.
It does not print credentials or environment variable values.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.supabase_client import fetch_match_data

REQUIRED_EXPORT_COLUMNS = (
    "match_date",
    "round_label",
    "home_team",
    "away_team",
    "home_xg",
    "away_xg",
    "home_np_xg",
    "away_np_xg",
    "home_goals",
    "away_goals",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export WSL match data from Supabase to CSV.")
    parser.add_argument(
        "--output",
        default="data/exports/wsl_match_data.csv",
        help="Local CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_match_data(Path(args.output))
    print_safe_summary(pd.read_csv(output_path), output_path)


def export_match_data(output_path: Path, *, df: pd.DataFrame | None = None) -> Path:
    """Fetch, validate, and export WSL match data to CSV."""
    data = fetch_match_data() if df is None else df.copy()
    validate_export_columns(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path


def validate_export_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required model-comparison columns are missing."""
    missing = [column for column in REQUIRED_EXPORT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Export data missing required columns: {missing}")


def safe_summary(df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Return a safe summary without secrets or row-level data."""
    validate_export_columns(df)
    dates = pd.to_datetime(df["match_date"], format="ISO8601", errors="coerce")
    round_labels = sorted(df["round_label"].dropna().astype(str).unique().tolist())
    return {
        "row_count": int(len(df)),
        "min_match_date": dates.min().date().isoformat() if not dates.empty and pd.notna(dates.min()) else None,
        "max_match_date": dates.max().date().isoformat() if not dates.empty and pd.notna(dates.max()) else None,
        "round_labels": round_labels,
        "output_path": str(output_path),
    }


def print_safe_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Print a concise, non-secret export summary."""
    summary = safe_summary(df, output_path)
    print("WSL match-data export complete")
    print(f"Rows: {summary['row_count']}")
    print(f"Date range: {summary['min_match_date']} to {summary['max_match_date']}")
    print(f"Round labels: {', '.join(summary['round_labels'])}")
    print(f"Output: {summary['output_path']}")


if __name__ == "__main__":
    main()

