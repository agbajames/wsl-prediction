#!/usr/bin/env python3
"""Normalise local upcoming fixture CSVs into the shadow fixture schema."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.shadow import normalise_fixture_frame

DATE_ALIASES = ("fixture_date", "match_date", "date")
HOME_ALIASES = ("home_team", "home", "home_team_name")
AWAY_ALIASES = ("away_team", "away", "away_team_name")
OPTIONAL_ALIASES: dict[str, tuple[str, ...]] = {
    "fixture_id": ("fixture_id", "id", "match_id", "game_id"),
    "round_label": ("round_label", "round", "matchweek", "match_week"),
    "season": ("season",),
    "competition": ("competition", "competition_name"),
    "venue": ("venue", "stadium"),
    "kickoff_time": ("kickoff_time", "kickoff", "time"),
    "source_notes": ("source_notes", "notes"),
}
OUTPUT_COLUMNS = (
    "fixture_id",
    "fixture_date",
    "round_label",
    "season",
    "home_team",
    "away_team",
    "competition",
    "venue",
    "kickoff_time",
    "source_notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalise a local upcoming-fixtures CSV into the canonical shadow fixture format."
    )
    parser.add_argument("--input", required=True, help="Raw local upcoming fixture CSV path.")
    parser.add_argument("--output", required=True, help="Normalised shadow fixture CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixtures = normalise_shadow_fixture_csv(Path(args.input))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(output_path, index=False)
    print(f"Wrote {len(fixtures)} normalised fixture row(s) to {output_path}.")


def normalise_shadow_fixture_csv(path: Path) -> pd.DataFrame:
    """Load a raw local fixture CSV and return canonical shadow fixture rows."""
    if path.suffix.lower() != ".csv":
        raise ValueError("Shadow fixture normalisation currently supports CSV input only.")
    raw = pd.read_csv(path)
    return normalise_shadow_fixture_frame(raw)


def normalise_shadow_fixture_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Return upcoming fixtures with canonical column names and strict validation."""
    source = raw.copy()
    if source.empty:
        raise ValueError("Fixture input is empty.")

    column_lookup = {_normalise_column_name(column): column for column in source.columns}
    mapped = pd.DataFrame(index=source.index)
    mapped["fixture_date"] = _required_column(source, column_lookup, DATE_ALIASES, "fixture date")
    mapped["home_team"] = _required_column(source, column_lookup, HOME_ALIASES, "home team")
    mapped["away_team"] = _required_column(source, column_lookup, AWAY_ALIASES, "away team")

    for canonical, aliases in OPTIONAL_ALIASES.items():
        column = _find_column(column_lookup, aliases)
        if column is not None:
            mapped[canonical] = source[column]

    try:
        normalized = normalise_fixture_frame(mapped)
    except ValueError as exc:
        if _looks_like_date_parse_error(str(exc)):
            raise ValueError("Fixture date must be parseable as ISO8601/date.") from exc
        raise
    _validate_no_blank_teams(normalized)
    _validate_distinct_teams(normalized)
    _validate_no_duplicate_fixtures(normalized)
    return _ordered_output(normalized)


def _required_column(
    source: pd.DataFrame,
    column_lookup: dict[str, str],
    aliases: tuple[str, ...],
    description: str,
) -> pd.Series:
    column = _find_column(column_lookup, aliases)
    if column is None:
        raise ValueError(f"Fixture input missing required {description} column. Accepted names: {aliases}.")
    values = source[column]
    if values.isna().any() or (values.astype(str).str.strip() == "").any():
        raise ValueError(f"Fixture input contains blank {description} values.")
    return values


def _find_column(column_lookup: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        column = column_lookup.get(_normalise_column_name(alias))
        if column is not None:
            return column
    return None


def _normalise_column_name(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_")


def _looks_like_date_parse_error(message: str) -> bool:
    return any(fragment in message for fragment in ("ISO8601", "time data", "DateParseError", "datetime"))


def _validate_no_blank_teams(fixtures: pd.DataFrame) -> None:
    for column in ("home_team", "away_team"):
        if fixtures[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Fixture input contains blank {column} values.")


def _validate_distinct_teams(fixtures: pd.DataFrame) -> None:
    same_team = fixtures["home_team"].str.casefold() == fixtures["away_team"].str.casefold()
    if same_team.any():
        row_numbers = [str(index + 2) for index in fixtures.index[same_team].tolist()]
        raise ValueError(f"Home and away teams cannot be identical. Check CSV row(s): {', '.join(row_numbers)}.")


def _validate_no_duplicate_fixtures(fixtures: pd.DataFrame) -> None:
    key_columns = ["fixture_date", "home_team", "away_team"]
    duplicates = fixtures.duplicated(subset=key_columns, keep=False)
    if duplicates.any():
        duplicate_rows = fixtures.loc[duplicates, key_columns]
        raise ValueError(f"Duplicate fixture rows found: {duplicate_rows.to_dict(orient='records')}.")

    fixture_ids = fixtures["fixture_id"].astype(str).str.strip()
    duplicate_ids = fixture_ids.ne("") & fixture_ids.duplicated(keep=False)
    if duplicate_ids.any():
        ids = sorted(fixture_ids.loc[duplicate_ids].unique())
        raise ValueError(f"Duplicate fixture_id values found: {ids}.")


def _ordered_output(fixtures: pd.DataFrame) -> pd.DataFrame:
    output = fixtures.copy()
    output["fixture_date"] = output["fixture_date"].dt.date.astype(str)
    output = output.drop(columns=["match_date"], errors="ignore")
    for column in OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    return output.loc[:, list(OUTPUT_COLUMNS)]


if __name__ == "__main__":
    main()
