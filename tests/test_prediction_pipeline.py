"""
tests/test_prediction_pipeline.py
----------------------------------
Unit tests for the WSL Prediction Engine.

Tests cover:
  - Data schema validation
  - Model output correctness
  - API endpoint behaviour (mocked Supabase)
  - Evaluation store (mocked Supabase)

Run with: pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.supabase_client import EXPECTED_COLS, fetch_match_data
from model.wsl_xg_model import (
    ModelConfig,
    compute_scoreline_matrix,
    estimate_penalty_rates,
    estimate_team_strengths,
    predict_fixtures,
    split_played_future,
    wdl_from_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal match DataFrame matching the RPC schema."""
    return pd.DataFrame(
        {
            "match_date": pd.to_datetime(
                [
                    "2025-09-07",
                    "2025-09-14",
                    "2025-09-21",
                    "2025-09-28",
                    "2025-10-05",
                    "2025-10-12",
                    "2025-11-01",
                    "2025-11-08",
                    "2025-11-15",
                ]
            ),
            "round_label": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"],
            "home_team": [
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Chelsea",
                "Arsenal",
            ],
            "away_team": [
                "Chelsea",
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Chelsea",
            ],
            "home_xg": [2.5, 1.2, 2.0, 1.8, 2.3, 1.5, 2.1, 1.9, 2.4],
            "away_xg": [1.0, 2.3, 1.1, 2.1, 0.9, 2.2, 1.0, 2.0, 1.1],
            "home_np_xg": [2.5, 1.2, 2.0, 1.8, 2.3, 1.5, 2.1, 1.9, 2.4],
            "away_np_xg": [1.0, 2.3, 1.1, 2.1, 0.9, 2.2, 1.0, 2.0, 1.1],
            "home_goals": [2, 1, 3, 1, 2, 0, 2, 1, 3],
            "away_goals": [0, 2, 0, 2, 0, 1, 1, 2, 0],
        }
    )


@pytest.fixture
def mocked_api_client(monkeypatch):
    """FastAPI test client with auth env vars set and Supabase startup mocked."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")

    with patch("api.main.get_supabase_client", return_value=MagicMock()):
        from api.main import app

        yield TestClient(app)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_expected_cols_match_required_cols(self):
        """EXPECTED_COLS in supabase_client must match REQUIRED_COLS in model."""
        from model.wsl_xg_model import REQUIRED_COLS

        assert EXPECTED_COLS == REQUIRED_COLS

    def test_sample_df_has_all_cols(self, sample_df):
        assert EXPECTED_COLS.issubset(set(sample_df.columns))

    def test_fetch_match_data_validates_mocked_rpc_schema(self, sample_df):
        """Supabase RPC rows are validated and coerced without live credentials."""
        rpc_rows = sample_df.assign(match_date=sample_df["match_date"].dt.strftime("%Y-%m-%d")).to_dict(
            orient="records"
        )
        supabase_client = MagicMock()
        supabase_client.rpc.return_value.execute.return_value = SimpleNamespace(data=rpc_rows)

        df = fetch_match_data(supabase_client)

        supabase_client.rpc.assert_called_once_with("rpc_wsl_weekly_stats")
        assert EXPECTED_COLS.issubset(df.columns)
        assert pd.api.types.is_datetime64_any_dtype(df["match_date"])
        assert pd.api.types.is_numeric_dtype(df["home_xg"])

    def test_fetch_match_data_rejects_missing_rpc_columns(self, sample_df):
        rpc_rows = sample_df.drop(columns=["home_np_xg"]).to_dict(orient="records")
        supabase_client = MagicMock()
        supabase_client.rpc.return_value.execute.return_value = SimpleNamespace(data=rpc_rows)

        with pytest.raises(ValueError, match="missing expected columns"):
            fetch_match_data(supabase_client)


# ---------------------------------------------------------------------------
# Data splitting tests
# ---------------------------------------------------------------------------

class TestSplitting:
    def test_split_returns_correct_counts(self, sample_df):
        played, future = split_played_future(
            sample_df,
            train_before=pd.Timestamp("2025-11-15"),
            predict_from=pd.Timestamp("2025-11-15"),
            predict_to=pd.Timestamp("2025-11-15"),
        )
        assert len(played) == 8
        assert len(future) == 1

    def test_split_raises_on_empty_train(self, sample_df):
        with pytest.raises(ValueError, match="No played matches"):
            split_played_future(
                sample_df,
                train_before=pd.Timestamp("2025-01-01"),
                predict_from=pd.Timestamp("2025-09-07"),
                predict_to=pd.Timestamp("2025-09-07"),
            )

    def test_split_raises_on_empty_predict_window(self, sample_df):
        with pytest.raises(ValueError, match="No fixtures"):
            split_played_future(
                sample_df,
                train_before=pd.Timestamp("2025-11-15"),
                predict_from=pd.Timestamp("2026-01-01"),
                predict_to=pd.Timestamp("2026-01-07"),
            )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModel:
    def test_strengths_sum_to_zero(self, sample_df):
        """Sum-to-zero constraint on attack/defence parameters."""
        config = ModelConfig()
        played = sample_df[sample_df["match_date"] < pd.Timestamp("2025-11-15")]
        strengths = estimate_team_strengths(played, config)

        atk_sum = sum(strengths.attack.values())
        def_sum = sum(strengths.defence.values())

        assert abs(atk_sum) < 1e-8, f"Attack parameters don't sum to zero: {atk_sum}"
        assert abs(def_sum) < 1e-8, f"Defence parameters don't sum to zero: {def_sum}"

    def test_scoreline_matrix_sums_to_one(self):
        matrix = compute_scoreline_matrix(1.5, 1.0, pen_home=0.05, pen_away=0.03, rho=-0.13)
        assert abs(matrix.sum() - 1.0) < 1e-6

    def test_wdl_sums_to_one(self):
        matrix = compute_scoreline_matrix(1.5, 1.0)
        hw, d, aw = wdl_from_matrix(matrix)
        assert abs(hw + d + aw - 1.0) < 1e-6

    def test_predict_returns_all_fixtures(self, sample_df):
        config = ModelConfig()
        played = sample_df[sample_df["match_date"] < pd.Timestamp("2025-11-15")]
        future = sample_df[sample_df["match_date"] == pd.Timestamp("2025-11-15")]

        strengths = estimate_team_strengths(played, config)
        home_pen, away_pen = estimate_penalty_rates(played, config)
        preds = predict_fixtures(future, strengths, home_pen, away_pen, config)

        assert len(preds) == len(future)

    def test_home_advantage_positive(self, sample_df):
        """Home advantage should be positive (home team expected to score more)."""
        config = ModelConfig()
        played = sample_df[sample_df["match_date"] < pd.Timestamp("2025-11-15")]
        strengths = estimate_team_strengths(played, config)
        assert strengths.home_advantage > 0


# ---------------------------------------------------------------------------
# API tests (mocked Supabase)
# ---------------------------------------------------------------------------

class TestAPI:
    def test_health_endpoint(self, mocked_api_client):
        resp = mocked_api_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_auth_fails_when_api_key_missing(self, mocked_api_client):
        resp = mocked_api_client.get("/strengths?train_before=2025-11-15")
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Invalid or missing API key."

    def test_auth_fails_when_api_key_is_wrong(self, mocked_api_client):
        resp = mocked_api_client.get(
            "/strengths?train_before=2025-11-15",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Invalid or missing API key."

    def test_predict_endpoint_smoke_with_mocked_data(self, mocked_api_client, sample_df):
        with (
            patch("api.main.fetch_match_data", return_value=sample_df),
            patch("api.main.log_prediction_run", return_value="run-123"),
        ):
            resp = mocked_api_client.post(
                "/predict",
                headers={"X-API-Key": "test-api-key"},
                json={
                    "train_before": "2025-11-15",
                    "predict_from": "2025-11-15",
                    "predict_to": "2025-11-15",
                },
            )

        body = resp.json()
        assert resp.status_code == 200
        assert body["run_id"] == "run-123"
        assert body["meta"]["train_matches"] == 8
        assert body["meta"]["predict_fixtures"] == 1
        assert len(body["predictions"]) == 1
        assert body["team_strengths"]

    def test_backtest_endpoint_smoke_with_mocked_data(self, mocked_api_client, sample_df):
        backtest_result = SimpleNamespace(
            n_matches=3,
            brier_score=0.2142,
            log_loss=0.8123,
            calibration_bins=[{"bin": "0.0-0.2", "count": 1}],
            per_match=[{"match_date": "2025-11-01", "actual": "home"}],
        )

        with (
            patch("api.main.fetch_match_data", return_value=sample_df),
            patch("api.main.run_backtest", return_value=backtest_result) as run_backtest_mock,
        ):
            resp = mocked_api_client.post(
                "/backtest",
                headers={"X-API-Key": "test-api-key"},
                json={"backtest_start": "2025-10-01"},
            )

        body = resp.json()
        assert resp.status_code == 200
        assert body["n_matches_evaluated"] == 3
        assert body["brier_score"] == 0.2142
        assert body["log_loss"] == 0.8123
        run_backtest_mock.assert_called_once()
