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
from dashboard.matchweek_manifest import available_matchweeks, available_seasons, get_matchweek_window


def test_matchweek_manifest_lookup_returns_dates():
    seasons = available_seasons()
    assert "2025-26" in seasons

    weeks = available_matchweeks("2025-26")
    assert weeks[0] == 1

    window = get_matchweek_window("2025-26", 1)
    assert window is not None
    assert window.train_before == "2025-09-05"
    assert window.predict_from == "2025-09-05"
    assert window.predict_to == "2025-09-08"
    assert window.run_trigger == "dashboard-season-2025-26-week-01"


def test_missing_matchweek_returns_none():
    assert get_matchweek_window("2025-26", 99) is None
    assert available_matchweeks("2099-00") == []


def test_build_predict_payload_uses_backend_schema():
    window = get_matchweek_window("2025-26", 2)
    assert window is not None

    payload = build_predict_payload(window)

    assert payload == {
        "train_before": "2025-09-12",
        "predict_from": "2025-09-12",
        "predict_to": "2025-09-15",
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
