from __future__ import annotations

import os

import streamlit as st

from dashboard.components.evaluation_panel import render_evaluation_panel
from dashboard.components.history_panel import render_history_panel
from dashboard.components.prediction_panel import render_prediction_panel
from dashboard.matchweek_manifest import available_matchweeks, available_seasons, get_matchweek_window


def main() -> None:
    st.set_page_config(page_title="WSL Prediction Operations Dashboard", layout="wide")
    st.title("WSL Prediction Operations Dashboard")

    default_base_url = os.environ.get("PREDICTION_API_BASE_URL", "http://localhost:8000")
    default_api_key = os.environ.get("API_KEY", "")

    with st.sidebar:
        st.header("Configuration")
        base_url = st.text_input("API base URL", value=default_base_url)
        api_key = st.text_input("API key", value=default_api_key, type="password")

        seasons = available_seasons()
        season = st.selectbox("Season", seasons, index=0)

        weeks = available_matchweeks(season)
        week = st.selectbox("Matchweek", weeks, index=0)

    window = get_matchweek_window(season, week)

    if window:
        st.caption(
            f"Selected {window.season} matchweek {window.week}: "
            f"{window.predict_from} to {window.predict_to}"
        )

    prediction_tab, history_tab, evaluation_tab = st.tabs(["Predictions", "History", "Evaluation"])

    with prediction_tab:
        render_prediction_panel(base_url, api_key, window)

    with history_tab:
        render_history_panel(base_url, api_key)

    with evaluation_tab:
        render_evaluation_panel(window)


if __name__ == "__main__":
    main()
