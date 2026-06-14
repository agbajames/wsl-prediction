from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from dashboard.api_client import (
    PredictionApiError,
    build_predict_payload,
    generate_predictions,
    get_prediction_history,
)
from dashboard.components.history_panel import infer_prediction_status, summarize_prediction_history, summarize_prediction_run
from dashboard.matchweek_manifest import (
    OPENING_MATCHWEEK_PRIOR_NOTE,
    MatchweekWindow,
    available_matchweeks,
    available_seasons,
    get_matchweek_window,
    matchweek_validation_messages,
)


def test_matchweek_manifest_lookup_returns_dates():
    seasons = available_seasons()
    assert "2025-26" in seasons

    weeks = available_matchweeks("2025-26")
    assert weeks[0] == 1

    window = get_matchweek_window("2025-26", 1)
    assert window is not None
    assert window.train_before == "2025-09-05"
    assert window.predict_from == "2025-09-05"
    assert window.predict_to == "2025-09-07"
    assert window.run_trigger == "dashboard-season-2025-26-week-01"
    assert window.verified is True
    assert window.round_label == "R1"
    assert window.fixture_count == 6


def test_full_manifest_returns_22_matchweeks():
    assert available_matchweeks("2025-26") == list(range(1, 23))


def test_missing_matchweek_returns_none():
    assert get_matchweek_window("2025-26", 99) is None
    assert available_matchweeks("2099-00") == []


def test_verified_matchweeks_do_not_show_unverified_warning():
    window = get_matchweek_window("2025-26", 2)
    assert window is not None

    messages = matchweek_validation_messages(window)

    assert not any("not verified" in message for message in messages)


def test_unverified_matchweek_warning_logic():
    window = MatchweekWindow(
        season="2025-26",
        week=8,
        train_before="2025-11-08",
        predict_from="2025-11-08",
        predict_to="2025-11-09",
        verified=False,
    )

    messages = matchweek_validation_messages(window)

    assert "not verified" in messages[0]


def test_verified_manifest_dates_match_supabase_derived_windows():
    expected_windows = {
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

    for week, (predict_from, predict_to) in expected_windows.items():
        window = get_matchweek_window("2025-26", week)
        assert window is not None
        assert window.predict_from == predict_from
        assert window.predict_to == predict_to
        assert window.train_before == predict_from
        assert window.verified is True
        assert window.round_label == f"R{week}"
        assert window.fixture_count == 6


def test_irregular_rescheduled_rounds_are_flagged():
    irregular_weeks = {3, 14, 16, 20, 21}

    for week in irregular_weeks:
        window = get_matchweek_window("2025-26", week)
        assert window is not None
        assert window.status == "verified-rescheduled"
        assert "rescheduled" in window.note


def test_missing_matchweek_date_warning_logic():
    window = MatchweekWindow(
        season="2025-26",
        week=8,
        train_before="",
        predict_from="2025-10-31",
        predict_to="2025-11-03",
    )

    messages = matchweek_validation_messages(window)

    assert any("dates are incomplete" in message for message in messages)


def test_matchweek_1_historical_prior_note_logic():
    window = get_matchweek_window("2025-26", 1)
    assert window is not None

    messages = matchweek_validation_messages(window)

    assert OPENING_MATCHWEEK_PRIOR_NOTE in messages


def test_build_predict_payload_uses_backend_schema():
    window = get_matchweek_window("2025-26", 2)
    assert window is not None

    payload = build_predict_payload(window)

    assert payload == {
        "train_before": "2025-09-12",
        "predict_from": "2025-09-12",
        "predict_to": "2025-09-14",
        "run_trigger": "dashboard-season-2025-26-week-02",
    }


def test_generate_predictions_posts_expected_payload(monkeypatch):
    window = get_matchweek_window("2025-26", 1)
    assert window is not None

    response = httpx.Response(
        status_code=200,
        json={
            "run_id": "run-123",
            "meta": {"train_matches": 12, "predict_fixtures": 6, "rho_used": -0.13},
            "predictions": [],
            "team_strengths": [],
        },
    )
    post_mock = MagicMock(return_value=response)
    monkeypatch.setattr("dashboard.api_client.httpx.post", post_mock)

    result = generate_predictions(base_url="http://localhost:8000/", api_key="test-key", window=window)

    assert result["run_id"] == "run-123"
    post_mock.assert_called_once()
    call_kwargs = post_mock.call_args.kwargs
    assert post_mock.call_args.args[0] == "http://localhost:8000/predict"
    assert call_kwargs["headers"]["X-API-Key"] == "test-key"
    assert call_kwargs["json"]["run_trigger"] == "dashboard-season-2025-26-week-01"


def test_generate_predictions_raises_friendly_auth_error(monkeypatch):
    window = get_matchweek_window("2025-26", 1)
    assert window is not None

    monkeypatch.setattr("dashboard.api_client.httpx.post", MagicMock(return_value=httpx.Response(status_code=403)))

    with pytest.raises(PredictionApiError, match="API key"):
        generate_predictions(base_url="http://localhost:8000", api_key="wrong", window=window)


def test_get_prediction_history_returns_runs(monkeypatch):
    get_mock = MagicMock(return_value=httpx.Response(status_code=200, json={"runs": [{"run_id": "run-123"}]}))
    monkeypatch.setattr("dashboard.api_client.httpx.get", get_mock)

    runs = get_prediction_history(base_url="http://localhost:8000", api_key="test-key", n=5)

    assert runs == [{"run_id": "run-123"}]
    assert get_mock.call_args.args[0] == "http://localhost:8000/history"
    assert get_mock.call_args.kwargs["params"] == {"n": 5}


def test_history_summary_flattens_nested_prediction_data():
    run = {
        "id": "run-123",
        "created_at": "2026-06-14T10:00:00Z",
        "train_before": "2025-10-03",
        "predict_from": "2025-10-03",
        "predict_to": "2025-10-05",
        "run_trigger": "dashboard-season-2025-26-week-05",
        "predictions": [{"home_team": "Arsenal"}, {"home_team": "Chelsea"}],
        "team_strengths": [{"team": "Arsenal"}],
    }

    summary = summarize_prediction_run(run)

    assert summary == {
        "run_id": "run-123",
        "created_at": "2026-06-14T10:00:00Z",
        "train_before": "2025-10-03",
        "predict_from": "2025-10-03",
        "predict_to": "2025-10-05",
        "fixture_count": 2,
        "run_trigger": "dashboard-season-2025-26-week-05",
    }
    assert all(not isinstance(value, list | dict) for value in summary.values())


def test_history_summary_avoids_object_object_style_nested_output():
    summaries = summarize_prediction_history(
        [
            {
                "id": "run-123",
                "predict_from": "2025-10-03",
                "predict_to": "2025-10-06",
                "predictions": [{"nested": {"value": 1}}],
            }
        ]
    )

    assert summaries[0]["fixture_count"] == 1
    assert "predictions" not in summaries[0]


def test_prediction_status_inference_from_mocked_history():
    window = get_matchweek_window("2025-26", 5)
    assert window is not None

    status = infer_prediction_status(
        window,
        [{"predict_from": "2025-10-03", "predict_to": "2025-10-05", "id": "run-123"}],
    )

    assert status == "Predicted"


def test_prediction_status_inference_not_run_when_no_matching_history():
    window = get_matchweek_window("2025-26", 5)
    assert window is not None

    status = infer_prediction_status(window, [{"predict_from": "2026-01-01", "predict_to": "2026-01-04"}])

    assert status == "Not run"
