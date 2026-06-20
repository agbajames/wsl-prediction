"""
data/schemas.py
---------------
Canonical column groups and lightweight schema helpers for model evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

CHAMPION_REQUIRED_COLUMNS: tuple[str, ...] = (
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

EVALUATION_COLUMNS: tuple[str, ...] = (
    "actual_outcome",
    "prediction_timestamp",
    "model_version",
    "prediction_run_id",
    "p_home_win",
    "p_draw",
    "p_away_win",
    "scoreline_probabilities",
)

MARKET_COLUMNS: tuple[str, ...] = (
    "odds_source",
    "odds_snapshot_timestamp",
    "home_odds",
    "draw_odds",
    "away_odds",
    "odds_format",
    "raw_p_home",
    "raw_p_draw",
    "raw_p_away",
    "fair_p_home",
    "fair_p_draw",
    "fair_p_away",
    "odds_snapshot_label",
)

MULTI_LEAGUE_COLUMNS: tuple[str, ...] = (
    "season",
    "competition",
    "league",
    "match_id",
    "home_team_id",
    "away_team_id",
    "neutral_venue",
)


def missing_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> list[str]:
    """Return required columns absent from a DataFrame, preserving requested order."""
    existing = set(df.columns)
    return [column for column in required_columns if column not in existing]


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    context: str = "",
) -> None:
    """Raise ValueError if required columns are missing."""
    missing = missing_columns(df, required_columns)
    if not missing:
        return

    prefix = f"{context} " if context else ""
    raise ValueError(f"{prefix}missing required columns: {missing}")


def validate_date_coercion_ready(
    df: pd.DataFrame,
    date_columns: Iterable[str],
    *,
    context: str = "",
) -> None:
    """Check date columns can be parsed without mutating the input DataFrame."""
    validate_required_columns(df, date_columns, context=context)
    for column in date_columns:
        try:
            pd.to_datetime(df[column], format="ISO8601", errors="raise")
        except Exception as exc:
            prefix = f"{context} " if context else ""
            raise ValueError(f"{prefix}column {column!r} is not date-coercion ready.") from exc
