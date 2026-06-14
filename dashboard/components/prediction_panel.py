from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard.api_client import PredictionApiError, generate_predictions
from dashboard.matchweek_manifest import MatchweekWindow


def _show_dataframe(records: list[dict[str, Any]], *, empty_message: str) -> None:
    if not records:
        st.info(empty_message)
        return
    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)


def render_prediction_panel(base_url: str, api_key: str, window: MatchweekWindow | None) -> None:
    st.subheader("Prediction Generation")

    if window is None:
        st.warning("No matchweek dates are configured for this selection.")
        return

    st.caption(window.notes or "No notes for this matchweek.")
    st.json(
        {
            "train_before": window.train_before,
            "predict_from": window.predict_from,
            "predict_to": window.predict_to,
            "run_trigger": window.run_trigger,
        },
        expanded=False,
    )

    if not window.is_complete:
        st.warning("Matchweek dates are incomplete. Fill in train_before, predict_from, and predict_to first.")
        return

    if st.button("Generate Predictions", type="primary", disabled=not api_key):
        if not api_key:
            st.error("Enter an API key in the sidebar before generating predictions.")
            return

        with st.spinner("Calling prediction API..."):
            try:
                result = generate_predictions(base_url=base_url, api_key=api_key, window=window)
            except PredictionApiError as exc:
                st.error(str(exc))
                return

        meta = result.get("meta", {})
        st.success(f"Prediction run complete: {result.get('run_id') or 'not logged'}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Train Matches", meta.get("train_matches", 0))
        col2.metric("Predict Fixtures", meta.get("predict_fixtures", 0))
        col3.metric("Rho Used", meta.get("rho_used", "n/a"))

        st.markdown("#### Predictions")
        _show_dataframe(result.get("predictions", []), empty_message="The API returned no predictions.")

        st.markdown("#### Team Strengths")
        _show_dataframe(result.get("team_strengths", []), empty_message="The API returned no team strengths.")
