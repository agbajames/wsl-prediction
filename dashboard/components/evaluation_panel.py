from __future__ import annotations

import streamlit as st

from dashboard.matchweek_manifest import MatchweekWindow


def render_evaluation_panel(window: MatchweekWindow | None) -> None:
    st.subheader("Evaluation")

    if window is None or not window.is_complete:
        st.warning("Configure matchweek dates before preparing an evaluation command.")
        return

    evaluation_trigger = f"logged-replay-{window.season}-week-{window.week:02d}"
    matchweek_command = (
        "python -m evaluation.evaluate_logged_predictions "
        f"--season {window.season} "
        f"--week {window.week} "
        f"--run-trigger {evaluation_trigger}"
    )
    persisted_command = f"{matchweek_command} --persist"

    st.caption("Run locally after dashboard predictions have been logged to prediction_runs.")
    st.code(matchweek_command, language="bash")
    st.caption("Persist the evaluation result to Supabase when the evaluation_runs table is configured.")
    st.code(persisted_command, language="bash")
    st.info(
        "This evaluates logged dashboard predictions rather than re-running the model. "
        "Use the CLI for full Week 2-22 replay evaluation."
    )
    st.caption("Dashboard-triggered evaluation execution is intentionally deferred to a later branch.")
