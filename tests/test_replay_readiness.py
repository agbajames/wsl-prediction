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


DATA_DERIVED_WINDOWS = {
    1: ("2025-09-05", "2025-09-07"),
    2: ("2025-09-12", "2025-09-14"),
    3: ("2025-09-19", "2025-12-11"),
    4: ("2025-09-27", "2025-09-28"),
    5: ("2025-10-03", "2025-10-05"),
    6: ("2025-10-12", "2025-10-12"),
    7: ("2025-11-01", "2025-11-02"),
    8: ("2025-11-08", "2025-11-09"),
    9: ("2025-11-15", "2025-11-16"),
    10: ("2025-12-06", "2025-12-07"),
    11: ("2025-12-13", "2025-12-14"),
    12: ("2026-01-10", "2026-01-11"),
    13: ("2026-01-23", "2026-01-25"),
    14: ("2026-02-01", "2026-04-29"),
    15: ("2026-02-07", "2026-02-08"),
    16: ("2026-02-13", "2026-05-06"),
    17: ("2026-03-15", "2026-03-18"),
    18: ("2026-03-21", "2026-03-22"),
    19: ("2026-03-28", "2026-03-29"),
    20: ("2026-04-25", "2026-05-09"),
    21: ("2026-05-02", "2026-05-13"),
    22: ("2026-05-16", "2026-05-16"),
}


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


def _six_fixture_dates(start_date: str, end_date: str) -> list[str]:
    return [start_date, start_date, start_date, end_date, end_date, end_date]


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
            "manifest_predict_to": "2025-09-14",
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


def test_verified_manifest_matches_data_derived_windows_for_replay_target():
    rows = []
    for week, (start_date, end_date) in DATA_DERIVED_WINDOWS.items():
        rows.extend(_fixture_rows(f"R{week}", _six_fixture_dates(start_date, end_date)))

    windows = derive_fixture_windows(pd.DataFrame(rows))
    check = compare_manifest_to_fixture_windows(
        MATCHWEEK_MANIFEST["2025-26"],
        windows,
        start_week=REPLAY_START_WEEK,
        end_week=REPLAY_END_WEEK,
    )

    assert check.manifest_weeks_missing_from_data == []
    assert check.data_rounds_missing_from_manifest == []
    assert check.date_mismatches == []
    assert check.fixture_count_mismatches == []
    assert check.unverified_windows == []
    assert check.expected_fixture_count == 126
