from __future__ import annotations

import pandas as pd
import pytest

from data.schemas import (
    CHAMPION_REQUIRED_COLUMNS,
    EVALUATION_COLUMNS,
    MARKET_COLUMNS,
    MULTI_LEAGUE_COLUMNS,
    missing_columns,
    validate_date_coercion_ready,
    validate_required_columns,
)


def _champion_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_date": ["2025-09-12"],
            "round_label": ["R2"],
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "home_xg": [1.4],
            "away_xg": [0.9],
            "home_np_xg": [1.4],
            "away_np_xg": [0.9],
            "home_goals": [2],
            "away_goals": [1],
        }
    )


def test_valid_champion_dataframe_passes_required_column_validation():
    df = _champion_df()

    validate_required_columns(df, CHAMPION_REQUIRED_COLUMNS, context="champion model")

    assert missing_columns(df, CHAMPION_REQUIRED_COLUMNS) == []


def test_missing_required_columns_are_reported_clearly():
    df = _champion_df().drop(columns=["home_np_xg", "away_np_xg"])

    missing = missing_columns(df, CHAMPION_REQUIRED_COLUMNS)

    assert missing == ["home_np_xg", "away_np_xg"]


def test_validation_error_message_includes_context():
    df = _champion_df().drop(columns=["home_xg"])

    with pytest.raises(ValueError, match="champion model missing required columns: \\['home_xg'\\]"):
        validate_required_columns(df, CHAMPION_REQUIRED_COLUMNS, context="champion model")


def test_optional_column_groups_do_not_fail_champion_validation():
    df = _champion_df()

    validate_required_columns(df, CHAMPION_REQUIRED_COLUMNS)

    assert missing_columns(df, EVALUATION_COLUMNS)
    assert missing_columns(df, MARKET_COLUMNS)
    assert missing_columns(df, MULTI_LEAGUE_COLUMNS)


def test_schema_helpers_do_not_mutate_input_dataframe():
    df = _champion_df()
    before = df.copy(deep=True)

    validate_required_columns(df, CHAMPION_REQUIRED_COLUMNS)
    validate_date_coercion_ready(df, ["match_date"], context="champion model")

    pd.testing.assert_frame_equal(df, before)
    assert df["match_date"].dtype == before["match_date"].dtype


def test_date_coercion_readiness_reports_invalid_dates_with_context():
    df = _champion_df()
    df.loc[0, "match_date"] = "not-a-date"

    with pytest.raises(ValueError, match="champion model column 'match_date' is not date-coercion ready"):
        validate_date_coercion_ready(df, ["match_date"], context="champion model")
