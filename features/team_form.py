"""
Leakage-aware team-form features for challenger models.

Training features are built sequentially in match-date order, so each row uses
only matches played before that row. Prediction features use summaries fitted
on the supplied training fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_RESULT_COLUMNS = (
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)

FEATURE_NAMES = (
    "home_points_per_match",
    "away_points_per_match",
    "points_per_match_diff",
    "home_goals_for_per_match",
    "away_goals_for_per_match",
    "home_goals_against_per_match",
    "away_goals_against_per_match",
    "goal_diff_per_match_diff",
    "home_indicator",
)


@dataclass
class TeamFormFeatureBuilder:
    """Build simple team-form features from match results."""

    default_points_per_match: float = 1.0
    default_goals_for_per_match: float = 1.2
    default_goals_against_per_match: float = 1.2
    team_stats: dict[str, dict[str, float]] = field(default_factory=dict, init=False)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def fit(self, played: pd.DataFrame) -> TeamFormFeatureBuilder:
        """Fit team summaries on all supplied played rows."""
        self.team_stats = {}
        for row in _ordered_played_rows(played).itertuples(index=False):
            self._update(str(row.home_team), int(row.home_goals), int(row.away_goals))
            self._update(str(row.away_team), int(row.away_goals), int(row.home_goals))
        return self

    def transform_training(self, played: pd.DataFrame) -> np.ndarray:
        """Build leakage-aware training features in chronological order."""
        self.team_stats = {}
        rows: list[list[float]] = []
        for row in _ordered_played_rows(played).itertuples(index=False):
            home_team = str(row.home_team)
            away_team = str(row.away_team)
            rows.append(self._features_for_fixture(home_team, away_team))
            self._update(home_team, int(row.home_goals), int(row.away_goals))
            self._update(away_team, int(row.away_goals), int(row.home_goals))
        return np.asarray(rows, dtype=float)

    def transform_fixtures(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Build prediction features from already-fitted team summaries."""
        rows = [
            self._features_for_fixture(str(row.home_team), str(row.away_team))
            for row in fixtures.itertuples(index=False)
        ]
        return np.asarray(rows, dtype=float)

    def _features_for_fixture(self, home_team: str, away_team: str) -> list[float]:
        home = self._rates(home_team)
        away = self._rates(away_team)
        home_gd = home["goals_for"] - home["goals_against"]
        away_gd = away["goals_for"] - away["goals_against"]
        return [
            home["points"],
            away["points"],
            home["points"] - away["points"],
            home["goals_for"],
            away["goals_for"],
            home["goals_against"],
            away["goals_against"],
            home_gd - away_gd,
            1.0,
        ]

    def _rates(self, team: str) -> dict[str, float]:
        stats = self.team_stats.get(team)
        if not stats or stats["matches"] <= 0:
            return {
                "points": self.default_points_per_match,
                "goals_for": self.default_goals_for_per_match,
                "goals_against": self.default_goals_against_per_match,
            }

        matches = stats["matches"]
        return {
            "points": stats["points"] / matches,
            "goals_for": stats["goals_for"] / matches,
            "goals_against": stats["goals_against"] / matches,
        }

    def _update(self, team: str, goals_for: int, goals_against: int) -> None:
        stats = self.team_stats.setdefault(
            team,
            {"matches": 0.0, "points": 0.0, "goals_for": 0.0, "goals_against": 0.0},
        )
        stats["matches"] += 1.0
        stats["goals_for"] += float(goals_for)
        stats["goals_against"] += float(goals_against)
        if goals_for > goals_against:
            stats["points"] += 3.0
        elif goals_for == goals_against:
            stats["points"] += 1.0


def training_outcomes(played: pd.DataFrame) -> list[str]:
    """Return H/D/A outcomes in the same order as transform_training."""
    outcomes = []
    for row in _ordered_played_rows(played).itertuples(index=False):
        if int(row.home_goals) > int(row.away_goals):
            outcomes.append("H")
        elif int(row.home_goals) < int(row.away_goals):
            outcomes.append("A")
        else:
            outcomes.append("D")
    return outcomes


def validate_result_columns(df: pd.DataFrame) -> None:
    """Validate required result columns for team-form features."""
    missing = [column for column in REQUIRED_RESULT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required logistic feature columns: {missing}")


def _ordered_played_rows(df: pd.DataFrame) -> pd.DataFrame:
    validate_result_columns(df)
    played = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()
    played["match_date"] = pd.to_datetime(played["match_date"], format="ISO8601", errors="raise")
    return played.sort_values(["match_date", "home_team", "away_team"], kind="mergesort")
