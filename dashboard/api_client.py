"""
dashboard/api_client.py
-----------------------
Thin client for the FastAPI prediction engine.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import httpx

from dashboard.matchweek_manifest import MatchweekWindow


class PredictionApiError(RuntimeError):
    """Raised when the prediction API cannot satisfy a dashboard request."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def build_predict_payload(window: MatchweekWindow) -> dict[str, Any]:
    """Build the POST /predict payload for a selected matchweek."""
    payload = asdict(window)
    payload.pop("season")
    payload.pop("week")
    payload.pop("notes")
    payload["run_trigger"] = window.run_trigger
    return payload


def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key, "Content-Type": "application/json"}


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    if response.status_code == 403:
        raise PredictionApiError("API key was missing or rejected by the prediction API.", status_code=403)
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise PredictionApiError(f"Prediction API returned {response.status_code}: {detail}", response.status_code)


def generate_predictions(
    *,
    base_url: str,
    api_key: str,
    window: MatchweekWindow,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Call POST /predict for a selected matchweek."""
    payload = build_predict_payload(window)
    try:
        response = httpx.post(
            _api_url(base_url, "/predict"),
            headers=_headers(api_key),
            json=payload,
            timeout=timeout_seconds,
        )
    except httpx.RequestError as exc:
        raise PredictionApiError(f"Prediction API is not reachable at {base_url}: {exc}") from exc

    _raise_for_status(response)
    return response.json()


def get_prediction_history(
    *,
    base_url: str,
    api_key: str,
    n: int = 10,
    timeout_seconds: float = 15.0,
) -> list[dict[str, Any]]:
    """Call GET /history and return recent prediction runs."""
    try:
        response = httpx.get(
            _api_url(base_url, "/history"),
            headers=_headers(api_key),
            params={"n": n},
            timeout=timeout_seconds,
        )
    except httpx.RequestError as exc:
        raise PredictionApiError(f"Prediction API is not reachable at {base_url}: {exc}") from exc

    _raise_for_status(response)
    return response.json().get("runs", [])
