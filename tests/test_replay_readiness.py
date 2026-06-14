from __future__ import annotations

import pandas as pd

from dashboard.matchweek_manifest import MATCHWEEK_MANIFEST
from scripts.inspect_matchweek_windows import (
    EXPECTED_REPLAY_FIXTURES,
    REPLAY_END_WEEK,
    REPLAY_START_WEEK,
    compare_manifest_to_fixture_windows,
    derive_fixture_windows,
    extract_week_number,
)


def _fixture_rows(round_label: str, dates: list[str], *, completed: bool = True) -> list[dict]:
    rows = []
    for idx, match_date in enumerate(dates, start=1):
        rows.append(
            {
                "round_label": round_label,
                "match_date": match_date,
                "home_team": f"Home {idx}",
                "away_team": f"Away {idx}",
                "home_xg": 1.2 if completed else None,
                "away_xg": 0.8 if completed else None,
                "home_np_xg": 1.1 if completed else None,
                "away_np_xg": 0.7 if completed else None,
                "home_goals": 1 if completed else None,
                "away_goals": 0 if completed else None,
            }
        )
    return rows


def test_extract_week_number_supports_common_round_labels():
    assert extract_week_number("R2") == 2
    assert extract_week_number("Round 14") == 14
    assert extract_week_number("Matchweek 22") == 22
    assert extract_week_number("Final") is None


def test_deriving_matchweek_windows_from_mocked_fixture_data():
    df = pd.DataFrame(
        _fixture_rows("R2", ["2025-09-12", "2025-09-13", "2025-09-14", "2025-09-14", "2025-09-15", "2025-09-15"])
        + _fixture_rows("R3", ["2025-09-19", "2025-09-20", "2025-09-21"], completed=False)
    )

    windows = derive_fixture_windows(df)

    assert len(windows) == 2
    assert windows[0].round_label == "R2"
    assert windows[0].week == 2
    assert windows[0].min_match_date == "2025-09-12"
    assert windows[0].max_match_date == "2025-09-15"
    assert windows[0].fixture_count == 6
    assert windows[0].completed_count == 6
    assert windows[0].notes == ""
    assert windows[1].fixture_count == 3
    assert windows[1].completed_count == 0
    assert "Expected 6 fixtures" in windows[1].notes


def test_manifest_date_mismatch_detection():
    df = pd.DataFrame(
        _fixture_rows("R2", ["2025-09-13", "2025-09-13", "2025-09-14", "2025-09-14", "2025-09-15", "2025-09-15"])
    )
    windows = derive_fixture_windows(df)

    check = compare_manifest_to_fixture_windows(MATCHWEEK_MANIFEST["2025-26"], windows, start_week=2, end_week=2)

    assert check.date_mismatches == [
        {
            "week": 2,
            "manifest_predict_from": "2025-09-12",
            "manifest_predict_to": "2025-09-15",
            "data_min_match_date": "2025-09-13",
            "data_max_match_date": "2025-09-15",
        }
    ]


def test_fixture_count_anomaly_detection():
    df = pd.DataFrame(_fixture_rows("R2", ["2025-09-12", "2025-09-13", "2025-09-14", "2025-09-15", "2025-09-15"]))
    windows = derive_fixture_windows(df)

    check = compare_manifest_to_fixture_windows(MATCHWEEK_MANIFEST["2025-26"], windows, start_week=2, end_week=2)

    assert check.fixture_count_mismatches == [
        {
            "week": 2,
            "round_label": "R2",
            "expected_fixture_count": 6,
            "actual_fixture_count": 5,
        }
    ]


def test_manifest_and_data_missing_detection():
    df = pd.DataFrame(
        _fixture_rows("R2", ["2025-09-12", "2025-09-13", "2025-09-14", "2025-09-14", "2025-09-15", "2025-09-15"])
        + _fixture_rows("Cup Final", ["2025-12-20"])
    )
    windows = derive_fixture_windows(df)

    check = compare_manifest_to_fixture_windows(MATCHWEEK_MANIFEST["2025-26"], windows, start_week=2, end_week=3)

    assert check.manifest_weeks_missing_from_data == [3]
    assert check.data_rounds_missing_from_manifest == ["Cup Final"]


def test_week_2_to_22_replay_target_count_and_week_1_flag():
    check = compare_manifest_to_fixture_windows(
        MATCHWEEK_MANIFEST["2025-26"],
        [],
        start_week=REPLAY_START_WEEK,
        end_week=REPLAY_END_WEEK,
    )

    assert REPLAY_START_WEEK == 2
    assert REPLAY_END_WEEK == 22
    assert EXPECTED_REPLAY_FIXTURES == 126
    assert check.expected_fixture_count == 126
    assert check.week_1_historical_prior_required is True
