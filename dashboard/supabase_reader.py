"""
dashboard/supabase_reader.py
----------------------------
Optional Supabase read helpers for internal dashboard history panels.

The baseline dashboard prefers FastAPI for prediction generation and history.
These helpers are available for later direct-read panels when the required
environment variables are configured.
"""

from __future__ import annotations

import os
from typing import Any

from supabase import Client, create_client


def get_dashboard_supabase_client() -> Client | None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key or url == "placeholder" or key == "placeholder":
        return None
    return create_client(url, key)


def get_latest_evaluation_runs(client: Client, n: int = 10) -> list[dict[str, Any]]:
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
