"""
evaluation/eval_store.py
------------------------
Persists prediction runs and backtest metrics to Supabase.

Every time the API generates predictions, a record is written to the
`prediction_runs` table. This gives a full audit trail of:
  - What was predicted and when
  - Which model config was used
  - What the backtest metrics were at the time of prediction

Required Supabase table (run once in Supabase SQL editor):

    CREATE TABLE IF NOT EXISTS prediction_runs (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        train_before    DATE NOT NULL,
        predict_from    DATE NOT NULL,
        predict_to      DATE NOT NULL,
        model_config    JSONB NOT NULL,
        predictions     JSONB NOT NULL,
        team_strengths  JSONB,
        brier_score     FLOAT,
        log_loss        FLOAT,
        n_matches_eval  INTEGER,
        rho_fitted      FLOAT,
        run_trigger     TEXT DEFAULT 'api'   -- 'api' | 'scheduled' | 'manual'
    );

    -- Index for querying recent runs
    CREATE INDEX IF NOT EXISTS idx_prediction_runs_created
        ON prediction_runs (created_at DESC);
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
from supabase import Client

logger = logging.getLogger("wsl_prediction.eval_store")


def log_prediction_run(
    client: Client,
    train_before: date,
    predict_from: date,
    predict_to: date,
    model_config: dict[str, Any],
    predictions_df: pd.DataFrame,
    strengths_df: pd.DataFrame | None = None,
    brier_score: float | None = None,
    log_loss: float | None = None,
    n_matches_eval: int | None = None,
    rho_fitted: float | None = None,
    run_trigger: str = "api",
) -> str:
    """Write a prediction run record to Supabase.

    Args:
        client:          Supabase client.
        train_before:    Training cutoff date.
        predict_from:    Start of prediction window.
        predict_to:      End of prediction window.
        model_config:    ModelConfig as a dict (alpha, decay, rho, etc.).
        predictions_df:  DataFrame of predictions from predict_fixtures().
        strengths_df:    Optional team strengths DataFrame.
        brier_score:     Backtest Brier score (if backtest was run).
        log_loss:        Backtest log-loss.
        n_matches_eval:  Number of matches evaluated in backtest.
        rho_fitted:      The rho value actually used (fitted or fixed).
        run_trigger:     How the run was triggered ('api', 'scheduled', 'manual').

    Returns:
        The UUID of the inserted record.
    """
    record = {
        "train_before": train_before.isoformat(),
        "predict_from": predict_from.isoformat(),
        "predict_to": predict_to.isoformat(),
        "model_config": model_config,
        "predictions": predictions_df.to_dict(orient="records"),
        "team_strengths": strengths_df.to_dict(orient="records") if strengths_df is not None else None,
        "brier_score": brier_score,
        "log_loss": log_loss,
        "n_matches_eval": n_matches_eval,
        "rho_fitted": rho_fitted,
        "run_trigger": run_trigger,
    }

    try:
        response = client.table("prediction_runs").insert(record).execute()
        run_id = response.data[0]["id"]
        logger.info("Prediction run logged → id=%s", run_id)
        return run_id
    except Exception as exc:
        # Non-fatal: log the error but don't fail the prediction
        logger.error("Failed to log prediction run to Supabase: %s", exc)
        return ""


def get_latest_predictions(client: Client, n: int = 1) -> list[dict]:
    """Fetch the most recent n prediction run records.

    Useful for the /history endpoint and monitoring.
    """
    try:
        response = (
            client.table("prediction_runs")
            .select("id, created_at, predict_from, predict_to, predictions, brier_score, log_loss")
            .order("created_at", desc=True)
            .limit(n)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.error("Failed to fetch prediction history: %s", exc)
        return []
