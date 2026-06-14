from __future__ import annotations

import streamlit as st

from dashboard.matchweek_manifest import MatchweekWindow


def render_evaluation_panel(window: MatchweekWindow | None) -> None:
    st.subheader("Evaluation")

    if window is None or not window.is_complete:
        st.warning("Configure matchweek dates before preparing an evaluation command.")
        return

    evaluation_trigger = f"{window.run_trigger}-evaluation"
    matchweek_command = (
        "python -m evaluation.run_evaluation "
        f"--start-date {window.predict_from} "
        f"--run-trigger {evaluation_trigger}"
    )
    persisted_command = f"{matchweek_command} --persist"

    st.caption("Run locally from the repository root after the matchweek has played.")
    st.code(matchweek_command, language="bash")
    st.caption("Persist the evaluation result to Supabase when the evaluation_runs table is configured.")
    st.code(persisted_command, language="bash")
    st.info(
        "The current runner may evaluate from the selected start date onward. "
        "A tighter matchweek-only evaluation window is a future checkpoint."
    )
    st.caption("Dashboard-triggered evaluation execution is intentionally deferred to a later branch.")
