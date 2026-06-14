from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard.api_client import PredictionApiError, get_prediction_history
from dashboard.matchweek_manifest import MatchweekWindow


def _run_id(run: dict[str, Any]) -> str:
    return str(run.get("run_id") or run.get("id") or "")


def _fixture_count(run: dict[str, Any]) -> int | None:
    if "fixture_count" in run and run["fixture_count"] is not None:
        return int(run["fixture_count"])
    predictions = run.get("predictions")
    if isinstance(predictions, list):
        return len(predictions)
    return None


def summarize_prediction_run(run: dict[str, Any]) -> dict[str, Any]:
    """Flatten a prediction run for the dashboard history table."""
    return {
        "run_id": _run_id(run),
        "created_at": run.get("created_at", ""),
        "train_before": run.get("train_before", ""),
        "predict_from": run.get("predict_from", ""),
        "predict_to": run.get("predict_to", ""),
        "fixture_count": _fixture_count(run),
        "run_trigger": run.get("run_trigger", ""),
    }


def summarize_prediction_history(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [summarize_prediction_run(run) for run in runs]


def infer_prediction_status(window: MatchweekWindow | None, runs: list[dict[str, Any]] | None) -> str:
    if window is None or not window.is_complete:
        return "Unknown"
    if runs is None:
        return "Unknown"

    for run in runs:
        if str(run.get("predict_from", "")) == window.predict_from and str(run.get("predict_to", "")) == window.predict_to:
            return "Predicted"
    return "Not run"


def render_history_panel(base_url: str, api_key: str) -> None:
    st.subheader("Prediction History")

    n = st.slider("Recent runs", min_value=1, max_value=50, value=10)
    if st.button("Refresh History", disabled=not api_key):
        if not api_key:
            st.error("Enter an API key in the sidebar before loading history.")
            return

        with st.spinner("Loading prediction history..."):
            try:
                runs = get_prediction_history(base_url=base_url, api_key=api_key, n=n)
            except PredictionApiError as exc:
                st.error(str(exc))
                return

        if not runs:
            st.info("No prediction runs returned.")
            return

        st.dataframe(pd.DataFrame(summarize_prediction_history(runs)), use_container_width=True, hide_index=True)

        with st.expander("Run details JSON"):
            for run in runs:
                st.markdown(f"**{_run_id(run) or 'Unlogged run'}**")
                st.json(run, expanded=False)
