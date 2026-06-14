"""
dashboard/matchweek_manifest.py
--------------------------------
Local matchweek date manifest for analyst dashboard operations.

The 2025-26 entries below are aligned to Supabase-derived fixture windows from
rpc_wsl_weekly_stats(). Rounds with long windows include postponed/rescheduled
fixtures and should be interpreted carefully during replay analysis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchweekWindow:
    season: str
    week: int
    train_before: str
    predict_from: str
    predict_to: str
    status: str = "verified"
    note: str = ""
    verified: bool = True
    round_label: str = ""
    fixture_count: int = 6

    @property
    def notes(self) -> str:
        """Backward-compatible alias for the dashboard baseline."""
        return self.note

    @property
    def run_trigger(self) -> str:
        week_label = f"{self.week:02d}"
        return f"dashboard-season-{self.season}-week-{week_label}"

    @property
    def is_complete(self) -> bool:
        return bool(self.train_before and self.predict_from and self.predict_to)


OPENING_MATCHWEEK_PRIOR_NOTE = (
    "Opening matchweek requires historical priors or previous-season baseline because no current-season matches "
    "exist before the first fixture."
)


IRREGULAR_ROUNDS = {3, 14, 16, 20, 21}


def _verified_week(week: int, predict_from: str, predict_to: str) -> MatchweekWindow:
    note = "Supabase-derived fixture window verified from rpc_wsl_weekly_stats()."
    status = "verified"
    if week == 1:
        note = f"{OPENING_MATCHWEEK_PRIOR_NOTE} {note}"
    if week in IRREGULAR_ROUNDS:
        status = "verified-rescheduled"
        note = (
            "Supabase-derived fixture window includes postponed/rescheduled fixtures; "
            "interpret replay metrics for this round carefully."
        )
    return MatchweekWindow(
        season="2025-26",
        week=week,
        train_before=predict_from,
        predict_from=predict_from,
        predict_to=predict_to,
        status=status,
        note=note,
        verified=True,
        round_label=f"R{week}",
        fixture_count=6,
    )


MATCHWEEK_MANIFEST: dict[str, dict[int, MatchweekWindow]] = {
    "2025-26": {
        1: _verified_week(1, "2025-09-05", "2025-09-07"),
        2: _verified_week(2, "2025-09-12", "2025-09-14"),
        3: _verified_week(3, "2025-09-19", "2025-12-11"),
        4: _verified_week(4, "2025-09-27", "2025-09-28"),
        5: _verified_week(5, "2025-10-03", "2025-10-05"),
        6: _verified_week(6, "2025-10-12", "2025-10-12"),
        7: _verified_week(7, "2025-11-01", "2025-11-02"),
        8: _verified_week(8, "2025-11-08", "2025-11-09"),
        9: _verified_week(9, "2025-11-15", "2025-11-16"),
        10: _verified_week(10, "2025-12-06", "2025-12-07"),
        11: _verified_week(11, "2025-12-13", "2025-12-14"),
        12: _verified_week(12, "2026-01-10", "2026-01-11"),
        13: _verified_week(13, "2026-01-23", "2026-01-25"),
        14: _verified_week(14, "2026-02-01", "2026-04-29"),
        15: _verified_week(15, "2026-02-07", "2026-02-08"),
        16: _verified_week(16, "2026-02-13", "2026-05-06"),
        17: _verified_week(17, "2026-03-15", "2026-03-18"),
        18: _verified_week(18, "2026-03-21", "2026-03-22"),
        19: _verified_week(19, "2026-03-28", "2026-03-29"),
        20: _verified_week(20, "2026-04-25", "2026-05-09"),
        21: _verified_week(21, "2026-05-02", "2026-05-13"),
        22: _verified_week(22, "2026-05-16", "2026-05-16"),
    }
}


def available_seasons() -> list[str]:
    return sorted(MATCHWEEK_MANIFEST.keys(), reverse=True)


def available_matchweeks(season: str) -> list[int]:
    return sorted(MATCHWEEK_MANIFEST.get(season, {}).keys())


def get_matchweek_window(season: str, week: int) -> MatchweekWindow | None:
    return MATCHWEEK_MANIFEST.get(season, {}).get(week)


def matchweek_validation_messages(window: MatchweekWindow | None) -> list[str]:
    if window is None:
        return ["No matchweek dates are configured for this selection."]

    messages: list[str] = []
    if not window.verified:
        messages.append("This matchweek window is not verified against official WSL fixtures.")
    if not window.is_complete:
        messages.append("Matchweek dates are incomplete. Fill in train_before, predict_from, and predict_to first.")
    if requires_opening_matchweek_prior_note(window):
        messages.append(OPENING_MATCHWEEK_PRIOR_NOTE)
    return messages


def requires_opening_matchweek_prior_note(window: MatchweekWindow | None) -> bool:
    return bool(window and window.week == 1)
