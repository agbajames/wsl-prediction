"""
Leakage-aware team-form features for challenger models.

Training features are built sequentially in match-date order, so each row uses
only matches played before that row. Prediction features use summaries fitted
on the supplied training fold.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

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

IMPROVED_FEATURE_GROUPS = ("base", "form", "xg", "opponent", "full")
IMPROVED_BASE_FEATURE_NAMES = FEATURE_NAMES
IMPROVED_FORM_FEATURE_NAMES = (
    "home_recent_points_per_match",
    "away_recent_points_per_match",
    "recent_points_per_match_diff",
    "home_recent_goal_diff_per_match",
    "away_recent_goal_diff_per_match",
    "recent_goal_diff_per_match_diff",
    "home_matches_played",
    "away_matches_played",
    "match_count_diff",
)
IMPROVED_XG_FEATURE_NAMES = (
    "home_xg_for_per_match",
    "away_xg_for_per_match",
    "home_xg_against_per_match",
    "away_xg_against_per_match",
    "xg_diff_per_match_diff",
    "home_recent_xg_diff_per_match",
    "away_recent_xg_diff_per_match",
    "recent_xg_diff_per_match_diff",
)
IMPROVED_OPPONENT_FEATURE_NAMES = (
    "home_opponent_points_allowed_per_match",
    "away_opponent_points_allowed_per_match",
    "opponent_points_allowed_diff",
    "home_opponent_goal_diff_allowed_per_match",
    "away_opponent_goal_diff_allowed_per_match",
    "opponent_goal_diff_allowed_diff",
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


@dataclass
class ImprovedTeamFormFeatureBuilder:
    """Build richer leakage-aware team features for improved logistic models."""

    feature_group: str = "full"
    recent_window: int = 4
    default_points_per_match: float = 1.0
    default_goals_for_per_match: float = 1.2
    default_goals_against_per_match: float = 1.2
    default_xg_for_per_match: float = 1.2
    default_xg_against_per_match: float = 1.2
    team_stats: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.feature_group not in IMPROVED_FEATURE_GROUPS:
            raise ValueError(f"feature_group must be one of: {IMPROVED_FEATURE_GROUPS}")
        names = list(IMPROVED_BASE_FEATURE_NAMES)
        if self.feature_group in {"form", "full"}:
            names.extend(IMPROVED_FORM_FEATURE_NAMES)
        if self.feature_group in {"xg", "full"}:
            names.extend(IMPROVED_XG_FEATURE_NAMES)
        if self.feature_group in {"opponent", "full"}:
            names.extend(IMPROVED_OPPONENT_FEATURE_NAMES)
        return tuple(names)

    def fit(self, played: pd.DataFrame) -> ImprovedTeamFormFeatureBuilder:
        """Fit summaries on all supplied rows."""
        self.team_stats = {}
        for row in _ordered_played_rows(played).itertuples(index=False):
            self._update_from_row(row)
        return self

    def transform_training(self, played: pd.DataFrame) -> np.ndarray:
        """Build training features sequentially, before each row is observed."""
        self.team_stats = {}
        rows: list[list[float]] = []
        for row in _ordered_played_rows(played).itertuples(index=False):
            rows.append(self._features_for_fixture(str(row.home_team), str(row.away_team)))
            self._update_from_row(row)
        return np.asarray(rows, dtype=float)

    def transform_fixtures(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Build prediction features from fitted pre-fixture summaries."""
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
        features = [
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
        if self.feature_group in {"form", "full"}:
            features.extend(self._form_features(home, away))
        if self.feature_group in {"xg", "full"}:
            features.extend(self._xg_features(home, away))
        if self.feature_group in {"opponent", "full"}:
            features.extend(self._opponent_features(home, away))
        return features

    def _form_features(self, home: dict[str, float], away: dict[str, float]) -> list[float]:
        home_recent_gd = home["recent_goals_for"] - home["recent_goals_against"]
        away_recent_gd = away["recent_goals_for"] - away["recent_goals_against"]
        return [
            home["recent_points"],
            away["recent_points"],
            home["recent_points"] - away["recent_points"],
            home_recent_gd,
            away_recent_gd,
            home_recent_gd - away_recent_gd,
            home["matches"],
            away["matches"],
            home["matches"] - away["matches"],
        ]

    def _xg_features(self, home: dict[str, float], away: dict[str, float]) -> list[float]:
        home_xg_diff = home["xg_for"] - home["xg_against"]
        away_xg_diff = away["xg_for"] - away["xg_against"]
        home_recent_xg_diff = home["recent_xg_for"] - home["recent_xg_against"]
        away_recent_xg_diff = away["recent_xg_for"] - away["recent_xg_against"]
        return [
            home["xg_for"],
            away["xg_for"],
            home["xg_against"],
            away["xg_against"],
            home_xg_diff - away_xg_diff,
            home_recent_xg_diff,
            away_recent_xg_diff,
            home_recent_xg_diff - away_recent_xg_diff,
        ]

    def _opponent_features(self, home: dict[str, float], away: dict[str, float]) -> list[float]:
        return [
            home["points_against"],
            away["points_against"],
            home["points_against"] - away["points_against"],
            home["opponent_goal_diff"],
            away["opponent_goal_diff"],
            home["opponent_goal_diff"] - away["opponent_goal_diff"],
        ]

    def _rates(self, team: str) -> dict[str, float]:
        stats = self.team_stats.get(team)
        if not stats or stats["matches"] <= 0:
            return {
                "matches": 0.0,
                "points": self.default_points_per_match,
                "goals_for": self.default_goals_for_per_match,
                "goals_against": self.default_goals_against_per_match,
                "recent_points": self.default_points_per_match,
                "recent_goals_for": self.default_goals_for_per_match,
                "recent_goals_against": self.default_goals_against_per_match,
                "xg_for": self.default_xg_for_per_match,
                "xg_against": self.default_xg_against_per_match,
                "recent_xg_for": self.default_xg_for_per_match,
                "recent_xg_against": self.default_xg_against_per_match,
                "points_against": self.default_points_per_match,
                "opponent_goal_diff": 0.0,
            }
        matches = float(stats["matches"])
        recent = list(stats["recent"])
        recent_n = max(float(len(recent)), 1.0)
        return {
            "matches": matches,
            "points": stats["points"] / matches,
            "goals_for": stats["goals_for"] / matches,
            "goals_against": stats["goals_against"] / matches,
            "recent_points": sum(item["points"] for item in recent) / recent_n,
            "recent_goals_for": sum(item["goals_for"] for item in recent) / recent_n,
            "recent_goals_against": sum(item["goals_against"] for item in recent) / recent_n,
            "xg_for": stats["xg_for"] / matches,
            "xg_against": stats["xg_against"] / matches,
            "recent_xg_for": sum(item["xg_for"] for item in recent) / recent_n,
            "recent_xg_against": sum(item["xg_against"] for item in recent) / recent_n,
            "points_against": stats["points_against"] / matches,
            "opponent_goal_diff": stats["opponent_goal_diff"] / matches,
        }

    def _update_from_row(self, row: Any) -> None:
        home_goals = int(row.home_goals)
        away_goals = int(row.away_goals)
        home_points, away_points = _points_for_match(home_goals, away_goals)
        home_xg = _row_value(row, "home_np_xg", _row_value(row, "home_xg", float(home_goals)))
        away_xg = _row_value(row, "away_np_xg", _row_value(row, "away_xg", float(away_goals)))
        self._update_team(str(row.home_team), home_goals, away_goals, home_points, away_points, home_xg, away_xg)
        self._update_team(str(row.away_team), away_goals, home_goals, away_points, home_points, away_xg, home_xg)

    def _update_team(
        self,
        team: str,
        goals_for: int,
        goals_against: int,
        points: int,
        points_against: int,
        xg_for: float,
        xg_against: float,
    ) -> None:
        stats = self.team_stats.setdefault(
            team,
            {
                "matches": 0.0,
                "points": 0.0,
                "goals_for": 0.0,
                "goals_against": 0.0,
                "xg_for": 0.0,
                "xg_against": 0.0,
                "points_against": 0.0,
                "opponent_goal_diff": 0.0,
                "recent": deque(maxlen=self.recent_window),
            },
        )
        stats["matches"] += 1.0
        stats["points"] += float(points)
        stats["goals_for"] += float(goals_for)
        stats["goals_against"] += float(goals_against)
        stats["xg_for"] += float(xg_for)
        stats["xg_against"] += float(xg_against)
        stats["points_against"] += float(points_against)
        stats["opponent_goal_diff"] += float(goals_against - goals_for)
        stats["recent"].append(
            {
                "points": float(points),
                "goals_for": float(goals_for),
                "goals_against": float(goals_against),
                "xg_for": float(xg_for),
                "xg_against": float(xg_against),
            }
        )


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


def _points_for_match(home_goals: int, away_goals: int) -> tuple[int, int]:
    if home_goals > away_goals:
        return 3, 0
    if home_goals < away_goals:
        return 0, 3
    return 1, 1


def _row_value(row: Any, column: str, fallback: float) -> float:
    value = getattr(row, column, fallback)
    if pd.isna(value):
        return fallback
    return float(value)
