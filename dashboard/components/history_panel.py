from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.api_client import PredictionApiError, get_prediction_history


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

        st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)
