"""
api/main.py
-----------
WSL Prediction Engine — FastAPI application.

Endpoints:
    GET  /health              Liveness check
    GET  /ready               Readiness check (Supabase connectivity)
    POST /predict             Generate match predictions
    GET  /strengths           Current team strength table
    GET  /history             Recent prediction runs
    POST /backtest            Run walk-forward backtest on demand

Auth:
    All prediction endpoints require an API key header:
        X-API-Key: <value of API_KEY env var>

    In production, API_KEY is injected from Azure Key Vault.
    Azure Container Apps handles TLS termination.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# Add project root to path so model module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.supabase_client import fetch_match_data, get_supabase_client
from evaluation.eval_store import get_latest_predictions, log_prediction_run
from model.wsl_xg_model import (
    ModelConfig,
    bootstrap_predictions,
    estimate_penalty_rates,
    estimate_team_strengths,
    fit_rho,
    predict_fixtures,
    run_backtest,
    split_played_future,
    team_strength_summary,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("wsl_prediction.api")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    expected = os.environ.get("API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY environment variable not configured.",
        )
    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )
    return api_key


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

supabase_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase_client
    logger.info("Starting WSL Prediction Engine...")
    supabase_client = get_supabase_client()
    logger.info("Supabase client initialised.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WSL Prediction Engine",
    description="xG-driven Dixon-Coles match prediction API for the Women's Super League.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    train_before: date = Field(..., description="Train on matches strictly before this date (YYYY-MM-DD)")
    predict_from: date = Field(..., description="First date of prediction window (inclusive)")
    predict_to: date = Field(..., description="Last date of prediction window (inclusive)")
    alpha: float = Field(0.15, ge=0.0, description="Ridge regularisation strength")
    decay_days: float = Field(60.0, gt=0, description="Time-decay half-life in days")
    rho: float | None = Field(-0.13, description="Dixon-Coles ρ. Pass null to fit from data.")
    bootstrap_n: int = Field(0, ge=0, description="Bootstrap resamples for CIs (0 = disabled)")
    run_trigger: str = Field("api", description="How this run was triggered")


class BacktestRequest(BaseModel):
    backtest_start: date = Field(..., description="First date to evaluate")
    alpha: float = Field(0.15, ge=0.0)
    decay_days: float = Field(60.0, gt=0)
    fit_rho: bool = Field(False, description="Fit ρ each batch from data")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health():
    """Liveness check — always returns 200 if the process is running."""
    return {"status": "ok", "service": "wsl-prediction-engine"}


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness check — verifies Supabase connectivity."""
    try:
        fetch_match_data(supabase_client)
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Supabase not reachable: {exc}")


@app.post("/predict", tags=["predictions"])
def predict(
    req: PredictRequest,
    _: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """
    Generate match predictions for a given date window.

    Trains on all played matches before `train_before`, then predicts
    fixtures in `[predict_from, predict_to]`.

    Returns predictions, team strengths, and run metadata.
    Every successful run is logged to the `prediction_runs` table.
    """
    config = ModelConfig(
        alpha=req.alpha,
        decay_half_life_days=req.decay_days,
        rho=req.rho,
        bootstrap_n=req.bootstrap_n,
    )

    # Fetch data directly from Supabase
    try:
        df = fetch_match_data(supabase_client)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Data fetch failed: {exc}")

    # Split
    try:
        played, future = split_played_future(
            df,
            pd.Timestamp(req.train_before),
            pd.Timestamp(req.predict_from),
            pd.Timestamp(req.predict_to),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Fit model
    strengths = estimate_team_strengths(played, config)
    rho = fit_rho(played, strengths, config) if config.rho is None else config.rho
    home_pen_rates, away_pen_rates = estimate_penalty_rates(played, config)

    # Predict
    predictions = predict_fixtures(future, strengths, home_pen_rates, away_pen_rates, config, rho=rho)

    # Bootstrap CIs
    if config.bootstrap_n > 0:
        cis = bootstrap_predictions(played, future, config, rho=rho, n_resamples=config.bootstrap_n)
        for pred in predictions:
            key = (pred.home_team, pred.away_team)
            if key in cis:
                pred.ci_home_win = cis[key]["home_win"]
                pred.ci_draw = cis[key]["draw"]
                pred.ci_away_win = cis[key]["away_win"]

    pred_df = pd.DataFrame([p.to_dict() for p in predictions])
    strengths_df = team_strength_summary(strengths, home_pen_rates, away_pen_rates)

    # Log to Supabase
    run_id = log_prediction_run(
        client=supabase_client,
        train_before=req.train_before,
        predict_from=req.predict_from,
        predict_to=req.predict_to,
        model_config={
            "alpha": config.alpha,
            "decay_half_life_days": config.decay_half_life_days,
            "rho": rho,
            "bootstrap_n": config.bootstrap_n,
        },
        predictions_df=pred_df,
        strengths_df=strengths_df,
        rho_fitted=rho,
        run_trigger=req.run_trigger,
    )

    return {
        "run_id": run_id,
        "meta": {
            "train_matches": len(played),
            "predict_fixtures": len(future),
            "rho_used": round(rho, 4),
            "train_before": req.train_before.isoformat(),
            "predict_from": req.predict_from.isoformat(),
            "predict_to": req.predict_to.isoformat(),
        },
        "predictions": pred_df.to_dict(orient="records"),
        "team_strengths": strengths_df.to_dict(orient="records"),
    }


@app.get("/strengths", tags=["predictions"])
def strengths(
    train_before: date,
    _: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """
    Return current team attack/defence strength ratings.

    Trains on all matches before `train_before`.
    """
    config = ModelConfig()

    try:
        df = fetch_match_data(supabase_client)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Data fetch failed: {exc}")

    played = df[
        (df["match_date"] < pd.Timestamp(train_before))
        & df["home_np_xg"].notna()
        & df["away_np_xg"].notna()
    ]

    if played.empty:
        raise HTTPException(status_code=422, detail=f"No played matches before {train_before}")

    strengths_obj = estimate_team_strengths(played, config)
    home_pen_rates, away_pen_rates = estimate_penalty_rates(played, config)
    strengths_df = team_strength_summary(strengths_obj, home_pen_rates, away_pen_rates)

    return {
        "as_of": train_before.isoformat(),
        "team_strengths": strengths_df.to_dict(orient="records"),
    }


@app.post("/backtest", tags=["evaluation"])
def backtest(
    req: BacktestRequest,
    _: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """
    Run a walk-forward backtest on demand.

    Evaluates weekly prediction batches from `backtest_start`
    to the most recent played match.
    """
    config = ModelConfig(
        alpha=req.alpha,
        decay_half_life_days=req.decay_days,
        rho=None if req.fit_rho else -0.13,
    )

    try:
        df = fetch_match_data(supabase_client)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Data fetch failed: {exc}")

    bt = run_backtest(
        df,
        config,
        start_date=pd.Timestamp(req.backtest_start),
        fit_rho_each_batch=req.fit_rho,
    )

    return {
        "n_matches_evaluated": bt.n_matches,
        "brier_score": round(bt.brier_score, 4),
        "log_loss": round(bt.log_loss, 4),
        "calibration_bins": bt.calibration_bins,
        "per_match_results": bt.per_match,
    }


@app.get("/history", tags=["evaluation"])
def history(
    n: int = 10,
    _: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Fetch the most recent n prediction runs from the audit log."""
    records = get_latest_predictions(supabase_client, n=n)
    return {"runs": records}
