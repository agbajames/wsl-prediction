"""
scripts/inspect_matchweek_windows.py
------------------------------------
Inspect WSL fixture windows from the same Supabase RPC data used by the API.

Run locally:
    python scripts/inspect_matchweek_windows.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.matchweek_manifest import MATCHWEEK_MANIFEST, MatchweekWindow, requires_opening_matchweek_prior_note
from data.supabase_client import fetch_match_data, get_supabase_client

EXPECTED_FIXTURES_PER_MATCHWEEK = 6
REPLAY_START_WEEK = 2
REPLAY_END_WEEK = 22
EXPECTED_REPLAY_MATCHWEEKS = REPLAY_END_WEEK - REPLAY_START_WEEK + 1
EXPECTED_REPLAY_FIXTURES = EXPECTED_REPLAY_MATCHWEEKS * EXPECTED_FIXTURES_PER_MATCHWEEK


@dataclass(frozen=True)
class FixtureWindowSummary:
    round_label: str
    week: int | None
    min_match_date: str
    max_match_date: str
    fixture_count: int
    completed_count: int | None
    notes: str = ""


@dataclass(frozen=True)
class ManifestCheck:
    replay_start_week: int
    replay_end_week: int
    expected_fixture_count: int
    manifest_weeks_missing_from_data: list[int]
    data_rounds_missing_from_manifest: list[str]
    date_mismatches: list[dict[str, Any]]
    fixture_count_mismatches: list[dict[str, Any]]
    unverified_windows: list[int]
    week_1_historical_prior_required: bool

    @property
    def is_ready(self) -> bool:
        return not (
            self.manifest_weeks_missing_from_data
            or self.data_rounds_missing_from_manifest
            or self.date_mismatches
            or self.fixture_count_mismatches
            or self.unverified_windows
        )


def extract_week_number(round_label: str) -> int | None:
    """Extract a matchweek number from labels such as R2, Round 2, or Matchweek 2."""
    match = re.search(r"(\d+)", str(round_label))
    if not match:
        return None
    return int(match.group(1))


def _completed_count(group: pd.DataFrame) -> int | None:
    if {"home_goals", "away_goals"}.issubset(group.columns):
        return int((group["home_goals"].notna() & group["away_goals"].notna()).sum())
    if {"home_xg", "away_xg"}.issubset(group.columns):
        return int((group["home_xg"].notna() & group["away_xg"].notna()).sum())
    return None


def derive_fixture_windows(df: pd.DataFrame) -> list[FixtureWindowSummary]:
    """Summarise available fixture windows by round_label."""
    if df.empty:
        return []

    required = {"round_label", "match_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fixture data missing required columns: {sorted(missing)}")

    work = df.copy()
    work["match_date"] = pd.to_datetime(work["match_date"], format="ISO8601", errors="raise")

    summaries: list[FixtureWindowSummary] = []
    for round_label, group in work.groupby("round_label", dropna=False):
        label = str(round_label)
        fixture_count = int(len(group))
        notes = ""
        if fixture_count != EXPECTED_FIXTURES_PER_MATCHWEEK:
            notes = f"Expected {EXPECTED_FIXTURES_PER_MATCHWEEK} fixtures; found {fixture_count}."

        summaries.append(
            FixtureWindowSummary(
                round_label=label,
                week=extract_week_number(label),
                min_match_date=group["match_date"].min().date().isoformat(),
                max_match_date=group["match_date"].max().date().isoformat(),
                fixture_count=fixture_count,
                completed_count=_completed_count(group),
                notes=notes,
            )
        )

    return sorted(summaries, key=lambda item: (item.week is None, item.week or 0, item.round_label))


def compare_manifest_to_fixture_windows(
    manifest: dict[int, MatchweekWindow],
    fixture_windows: list[FixtureWindowSummary],
    *,
    start_week: int = REPLAY_START_WEEK,
    end_week: int = REPLAY_END_WEEK,
    expected_fixtures_per_week: int = EXPECTED_FIXTURES_PER_MATCHWEEK,
) -> ManifestCheck:
    """Compare manifest windows with data-derived round windows."""
    manifest_weeks = set(range(start_week, end_week + 1))
    data_by_week = {window.week: window for window in fixture_windows if window.week is not None}
    data_weeks = set(data_by_week)

    manifest_weeks_missing_from_data = sorted(manifest_weeks - data_weeks)
    data_rounds_missing_from_manifest = [
        window.round_label
        for window in fixture_windows
        if window.week is None or window.week not in manifest
    ]
    date_mismatches: list[dict[str, Any]] = []
    fixture_count_mismatches: list[dict[str, Any]] = []

    for week in sorted(manifest_weeks & data_weeks):
        manifest_window = manifest[week]
        data_window = data_by_week[week]
        if (
            manifest_window.predict_from != data_window.min_match_date
            or manifest_window.predict_to != data_window.max_match_date
        ):
            date_mismatches.append(
                {
                    "week": week,
                    "manifest_predict_from": manifest_window.predict_from,
                    "manifest_predict_to": manifest_window.predict_to,
                    "data_min_match_date": data_window.min_match_date,
                    "data_max_match_date": data_window.max_match_date,
                }
            )
        if data_window.fixture_count != expected_fixtures_per_week:
            fixture_count_mismatches.append(
                {
                    "week": week,
                    "round_label": data_window.round_label,
                    "expected_fixture_count": expected_fixtures_per_week,
                    "actual_fixture_count": data_window.fixture_count,
                }
            )

    unverified_windows = sorted(
        week for week in manifest_weeks if week in manifest and not manifest[week].verified
    )

    return ManifestCheck(
        replay_start_week=start_week,
        replay_end_week=end_week,
        expected_fixture_count=(end_week - start_week + 1) * expected_fixtures_per_week,
        manifest_weeks_missing_from_data=manifest_weeks_missing_from_data,
        data_rounds_missing_from_manifest=data_rounds_missing_from_manifest,
        date_mismatches=date_mismatches,
        fixture_count_mismatches=fixture_count_mismatches,
        unverified_windows=unverified_windows,
        week_1_historical_prior_required=requires_opening_matchweek_prior_note(manifest.get(1)),
    )


def render_markdown_report(
    fixture_windows: list[FixtureWindowSummary],
    manifest_check: ManifestCheck,
) -> str:
    """Render a replay-readiness report suitable for saving under reports/."""
    lines = [
        "# WSL 2025-26 Replay Manifest Check",
        "",
        "## Replay Scope",
        "",
        f"- Weeks intended for replay: {manifest_check.replay_start_week}-{manifest_check.replay_end_week}",
        "- Week 1 excluded from this baseline because it requires historical priors or a previous-season baseline.",
        f"- Expected replay fixture count: {manifest_check.expected_fixture_count} "
        f"({EXPECTED_REPLAY_MATCHWEEKS} matchweeks x {EXPECTED_FIXTURES_PER_MATCHWEEK} fixtures).",
        "- Monte Carlo simulation is deferred until baseline replay metrics are captured.",
        "",
        "## Data-Derived Fixture Windows",
        "",
        "| round_label | week | min_match_date | max_match_date | fixture_count | completed_count | notes |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]

    if fixture_windows:
        for window in fixture_windows:
            completed = "" if window.completed_count is None else str(window.completed_count)
            lines.append(
                f"| {window.round_label} | {window.week or ''} | {window.min_match_date} | "
                f"{window.max_match_date} | {window.fixture_count} | {completed} | {window.notes} |"
            )
    else:
        lines.append("| _No live fixture data loaded_ |  |  |  |  |  | Run the script with Supabase credentials. |")

    lines.extend(
        [
            "",
            "## Manifest Verification Findings",
            "",
            f"- Ready for replay: {'yes' if manifest_check.is_ready else 'no'}",
            f"- Manifest weeks missing from data: {manifest_check.manifest_weeks_missing_from_data or 'none'}",
            f"- Data rounds missing from manifest: {manifest_check.data_rounds_missing_from_manifest or 'none'}",
            f"- Date mismatches: {manifest_check.date_mismatches or 'none'}",
            f"- Fixture-count mismatches: {manifest_check.fixture_count_mismatches or 'none'}",
            f"- Unverified manifest windows: {manifest_check.unverified_windows or 'none'}",
            f"- Week 1 historical-prior required: {manifest_check.week_1_historical_prior_required}",
            "",
            "## Week-By-Week Replay Steps",
            "",
            "1. Run `python scripts/inspect_matchweek_windows.py --output reports/replay_manifest_check.md`.",
            "2. Verify Matchweeks 2-22 have six fixtures each and no date mismatches.",
            "3. Update `dashboard/matchweek_manifest.py` only with data-derived or officially verified dates.",
            "4. Start the API with `python -m uvicorn api.main:app --env-file .env`.",
            "5. Start Streamlit with "
            "`PREDICTION_API_BASE_URL=http://localhost:8000 API_KEY=$API_KEY python -m streamlit run dashboard/app.py`.",
            "6. In the dashboard, generate predictions for Matchweeks 2 through 22 in order.",
            "7. Confirm each run appears in prediction history and therefore in `prediction_runs`.",
            "8. Run evaluation after predictions are logged, using the dashboard-generated command or "
            "`python -m evaluation.run_evaluation --start-date <week_predict_from> --persist "
            "--run-trigger dashboard-season-2025-26-week-XX-evaluation`.",
        ]
    )

    return "\n".join(lines) + "\n"


def load_fixture_windows_from_supabase() -> list[FixtureWindowSummary]:
    return derive_fixture_windows(fetch_match_data(get_supabase_client()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect WSL fixture windows from Supabase RPC data.")
    parser.add_argument("--season", default="2025-26", help="Dashboard manifest season to compare.")
    parser.add_argument("--output", default=None, help="Optional markdown output path.")
    args = parser.parse_args()

    fixture_windows = load_fixture_windows_from_supabase()
    manifest = MATCHWEEK_MANIFEST.get(args.season, {})
    manifest_check = compare_manifest_to_fixture_windows(manifest, fixture_windows)
    report = render_markdown_report(fixture_windows, manifest_check)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(report)
    print(report)


if __name__ == "__main__":
    main()
