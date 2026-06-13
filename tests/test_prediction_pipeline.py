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

import sys
import os
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.wsl_xg_model import (
    ModelConfig,
    estimate_penalty_rates,
    estimate_team_strengths,
    predict_fixtures,
    split_played_future,
    wdl_from_matrix,
    compute_scoreline_matrix,
)
from data.supabase_client import EXPECTED_COLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal match DataFrame matching the RPC schema."""
    return pd.DataFrame({
        "match_date": pd.to_datetime([
            "2025-09-07", "2025-09-14", "2025-09-21",
            "2025-09-28", "2025-10-05", "2025-10-12",
            "2025-11-01", "2025-11-08", "2025-11-15",
        ]),
        "round_label": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"],
        "home_team": ["Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal"],
        "away_team": ["Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea"],
        "home_xg":    [2.5, 1.2, 2.0, 1.8, 2.3, 1.5, 2.1, 1.9, 2.4],
        "away_xg":    [1.0, 2.3, 1.1, 2.1, 0.9, 2.2, 1.0, 2.0, 1.1],
        "home_np_xg": [2.5, 1.2, 2.0, 1.8, 2.3, 1.5, 2.1, 1.9, 2.4],
        "away_np_xg": [1.0, 2.3, 1.1, 2.1, 0.9, 2.2, 1.0, 2.0, 1.1],
        "home_goals": [2, 1, 3, 1, 2, 0, 2, 1, 3],
        "away_goals": [0, 2, 0, 2, 0, 1, 1, 2, 0],
    })


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
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient

        with patch("api.main.get_supabase_client") as mock_client:
            mock_client.return_value = MagicMock()

            from api.main import app
            client = TestClient(app)

            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
