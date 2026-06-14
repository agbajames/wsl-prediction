"""
dashboard/matchweek_manifest.py
--------------------------------
Local matchweek date manifest for analyst dashboard operations.

The first baseline includes editable placeholders for early 2025-26 WSL
matchweeks. Fill in the remaining weeks as fixture dates are confirmed.
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
    notes: str = ""

    @property
    def run_trigger(self) -> str:
        week_label = f"{self.week:02d}"
        return f"dashboard-season-{self.season}-week-{week_label}"

    @property
    def is_complete(self) -> bool:
        return bool(self.train_before and self.predict_from and self.predict_to)


MATCHWEEK_MANIFEST: dict[str, dict[int, MatchweekWindow]] = {
    "2025-26": {
        1: MatchweekWindow(
            season="2025-26",
            week=1,
            train_before="2025-09-05",
            predict_from="2025-09-05",
            predict_to="2025-09-08",
            notes="Placeholder opening matchweek window; verify against official fixtures.",
        ),
        2: MatchweekWindow(
            season="2025-26",
            week=2,
            train_before="2025-09-12",
            predict_from="2025-09-12",
            predict_to="2025-09-15",
            notes="Placeholder window.",
        ),
        3: MatchweekWindow(
            season="2025-26",
            week=3,
            train_before="2025-09-19",
            predict_from="2025-09-19",
            predict_to="2025-09-22",
            notes="Placeholder window.",
        ),
        4: MatchweekWindow(
            season="2025-26",
            week=4,
            train_before="2025-09-26",
            predict_from="2025-09-26",
            predict_to="2025-09-29",
            notes="Placeholder window.",
        ),
        5: MatchweekWindow(
            season="2025-26",
            week=5,
            train_before="2025-10-03",
            predict_from="2025-10-03",
            predict_to="2025-10-06",
            notes="Placeholder window.",
        ),
    }
}


def available_seasons() -> list[str]:
    return sorted(MATCHWEEK_MANIFEST.keys(), reverse=True)


def available_matchweeks(season: str) -> list[int]:
    return sorted(MATCHWEEK_MANIFEST.get(season, {}).keys())


def get_matchweek_window(season: str, week: int) -> MatchweekWindow | None:
    return MATCHWEEK_MANIFEST.get(season, {}).get(week)
