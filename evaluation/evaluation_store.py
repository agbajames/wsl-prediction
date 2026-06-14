"""
evaluation/evaluation_store.py
------------------------------
Persistence helpers for offline evaluation runs.

This module intentionally writes to `evaluation_runs`, not `prediction_runs`.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from supabase import Client

logger = logging.getLogger("wsl_prediction.evaluation_store")


def _as_iso_date(value: str | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def build_evaluation_run_record(
    *,
    evaluation_result: dict[str, Any],
    run_trigger: str = "manual",
    code_version: str | None = None,
    notes: str | None = None,
    data_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Supabase record for an evaluation result."""
    params = evaluation_result.get("parameters", {})
    metrics = evaluation_result.get("metrics", {})
    model_config = {
        "alpha": params.get("alpha"),
        "decay_days": params.get("decay_days"),
        "rho": params.get("rho"),
        "fit_rho_each_batch": params.get("fit_rho_each_batch"),
    }

    return {
        "evaluation_type": evaluation_result.get("evaluation_type", "walk_forward"),
        "start_date": _as_iso_date(params.get("start_date")),
        "end_date": _as_iso_date(params.get("end_date")),
        "model_config": model_config,
        "evaluation_params": params,
        "aggregate_metrics": {
            "n_matches": metrics.get("n_matches", 0),
            "brier_score": metrics.get("brier_score"),
            "log_loss": metrics.get("log_loss"),
            "accuracy": metrics.get("accuracy"),
        },
        "calibration_bins": metrics.get("calibration_bins", []),
        "confidence_buckets": metrics.get("confidence_buckets", []),
        "per_match_results": evaluation_result.get("per_match_results", []),
        "data_snapshot": data_snapshot,
        "run_trigger": run_trigger,
        "code_version": code_version,
        "notes": notes,
    }


def log_evaluation_run(
    client: Client,
    evaluation_result: dict[str, Any],
    *,
    run_trigger: str = "manual",
    code_version: str | None = None,
    notes: str | None = None,
    data_snapshot: dict[str, Any] | None = None,
) -> str:
    """Persist an evaluation result and return its run_id.

    Insert failures are logged and return an empty string so offline evaluation
    output remains available to the caller.
    """
    record = build_evaluation_run_record(
        evaluation_result=evaluation_result,
        run_trigger=run_trigger,
        code_version=code_version,
        notes=notes,
        data_snapshot=data_snapshot,
    )
    try:
        response = client.table("evaluation_runs").insert(record).execute()
        run_id = response.data[0]["run_id"]
        logger.info("Evaluation run logged -> run_id=%s", run_id)
        return run_id
    except Exception as exc:
        logger.error("Failed to log evaluation run to Supabase: %s", exc)
        return ""


def get_latest_evaluation_runs(client: Client, n: int = 10) -> list[dict[str, Any]]:
    """Fetch the most recent evaluation run records."""
    try:
        response = (
            client.table("evaluation_runs")
            .select(
                "run_id, created_at, evaluation_type, start_date, end_date, "
                "aggregate_metrics, run_trigger, code_version, notes"
            )
            .order("created_at", desc=True)
            .limit(n)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.error("Failed to fetch evaluation runs: %s", exc)
        return []
