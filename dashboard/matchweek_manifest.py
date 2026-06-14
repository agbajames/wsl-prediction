"""
dashboard/matchweek_manifest.py
--------------------------------
Local matchweek date manifest for analyst dashboard operations.

The 2025-26 entries below are editable placeholders. Verify every date against
the official WSL fixture list before using this manifest for final replay.
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
    status: str = "placeholder"
    note: str = ""
    verified: bool = False

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


def _placeholder_week(week: int, train_before: str, predict_from: str, predict_to: str) -> MatchweekWindow:
    note = "Placeholder replay window; verify against official WSL fixtures before final replay."
    if week == 1:
        note = f"{OPENING_MATCHWEEK_PRIOR_NOTE} {note}"
    return MatchweekWindow(
        season="2025-26",
        week=week,
        train_before=train_before,
        predict_from=predict_from,
        predict_to=predict_to,
        status="placeholder",
        note=note,
        verified=False,
    )


MATCHWEEK_MANIFEST: dict[str, dict[int, MatchweekWindow]] = {
    "2025-26": {
        1: _placeholder_week(1, "2025-09-05", "2025-09-05", "2025-09-08"),
        2: _placeholder_week(2, "2025-09-12", "2025-09-12", "2025-09-15"),
        3: _placeholder_week(3, "2025-09-19", "2025-09-19", "2025-09-22"),
        4: _placeholder_week(4, "2025-09-26", "2025-09-26", "2025-09-29"),
        5: _placeholder_week(5, "2025-10-03", "2025-10-03", "2025-10-06"),
        6: _placeholder_week(6, "2025-10-10", "2025-10-10", "2025-10-13"),
        7: _placeholder_week(7, "2025-10-17", "2025-10-17", "2025-10-20"),
        8: _placeholder_week(8, "2025-10-31", "2025-10-31", "2025-11-03"),
        9: _placeholder_week(9, "2025-11-07", "2025-11-07", "2025-11-10"),
        10: _placeholder_week(10, "2025-11-14", "2025-11-14", "2025-11-17"),
        11: _placeholder_week(11, "2025-11-21", "2025-11-21", "2025-11-24"),
        12: _placeholder_week(12, "2025-12-05", "2025-12-05", "2025-12-08"),
        13: _placeholder_week(13, "2025-12-12", "2025-12-12", "2025-12-15"),
        14: _placeholder_week(14, "2026-01-09", "2026-01-09", "2026-01-12"),
        15: _placeholder_week(15, "2026-01-16", "2026-01-16", "2026-01-19"),
        16: _placeholder_week(16, "2026-01-23", "2026-01-23", "2026-01-26"),
        17: _placeholder_week(17, "2026-02-06", "2026-02-06", "2026-02-09"),
        18: _placeholder_week(18, "2026-02-13", "2026-02-13", "2026-02-16"),
        19: _placeholder_week(19, "2026-03-06", "2026-03-06", "2026-03-09"),
        20: _placeholder_week(20, "2026-03-20", "2026-03-20", "2026-03-23"),
        21: _placeholder_week(21, "2026-04-24", "2026-04-24", "2026-04-27"),
        22: _placeholder_week(22, "2026-05-08", "2026-05-08", "2026-05-11"),
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
