"""
data/supabase_client.py
-----------------------
Direct Supabase data layer for the WSL prediction pipeline.

Replaces the manual workflow of:
    Supabase → export CSV → save locally → pass --csv path

Instead calls rpc_wsl_weekly_stats() directly and returns a DataFrame
with the exact schema expected by wsl_xg_model.load_and_validate().

Environment variables (set in Azure Key Vault / local .env):
    SUPABASE_URL              — e.g. https://xyzxyz.supabase.co
    SUPABASE_SERVICE_ROLE_KEY — service role key (never anon key in production)
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from supabase import create_client, Client

logger = logging.getLogger("wsl_prediction.data")

# Columns returned by rpc_wsl_weekly_stats() — must match model REQUIRED_COLS
EXPECTED_COLS = {
    "match_date",
    "round_label",
    "home_team",
    "away_team",
    "home_xg",
    "away_xg",
    "home_np_xg",
    "away_np_xg",
    "home_goals",
    "away_goals",
}


def get_supabase_client() -> Client:
    """Initialise Supabase client from environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set. "
            "In production these are injected from Azure Key Vault. "
            "Locally, add them to your .env file."
        )

    return create_client(url, key)


def fetch_match_data(client: Client | None = None) -> pd.DataFrame:
    """Call rpc_wsl_weekly_stats() and return a validated DataFrame.

    This is a drop-in replacement for load_and_validate(csv_path) —
    it returns the same schema so the model runs unchanged.

    Args:
        client: Optional pre-initialised Supabase client. If None,
                one is created from environment variables.

    Returns:
        DataFrame with columns matching REQUIRED_COLS in wsl_xg_model.py.

    Raises:
        EnvironmentError: If Supabase credentials are missing.
        ValueError: If the RPC returns unexpected or missing columns.
        RuntimeError: If the Supabase call fails.
    """
    if client is None:
        client = get_supabase_client()

    logger.info("Fetching match data from Supabase rpc_wsl_weekly_stats()...")

    try:
        response = client.rpc("rpc_wsl_weekly_stats").execute()
    except Exception as exc:
        raise RuntimeError(f"Supabase RPC call failed: {exc}") from exc

    if not response.data:
        raise ValueError("rpc_wsl_weekly_stats() returned no rows.")

    df = pd.DataFrame(response.data)

    # Validate columns
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"RPC response missing expected columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}"
        )

    # Parse dates (Supabase returns ISO8601 strings)
    df["match_date"] = pd.to_datetime(df["match_date"], format="ISO8601", errors="raise")

    # Preserve round_label as string
    df["round_label"] = df["round_label"].astype(str).where(df["round_label"].notna(), None)

    # Coerce numeric columns
    for col in ["home_xg", "away_xg", "home_np_xg", "away_np_xg", "home_goals", "away_goals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Fetched %d rows from Supabase (seasons: %s to %s)",
                len(df),
                df["match_date"].min().date(),
                df["match_date"].max().date())

    return df
