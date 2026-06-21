"""
Simple baseline challenger models for evaluation.

These models are intentionally transparent sanity checks. They implement the
``EvaluationModel`` protocol and operate only on local DataFrames.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = (
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)


@dataclass
class NaiveOutcomeRateBaseline:
    """Predict historical home/draw/away frequencies for every fixture."""

    prior_home: float = 1 / 3
    prior_draw: float = 1 / 3
    prior_away: float = 1 / 3
    prior_strength: float = 3.0

    _probabilities: tuple[float, float, float] = field(default=(1 / 3, 1 / 3, 1 / 3), init=False)
    _training_matches: int = field(default=0, init=False)

    @property
    def name(self) -> str:
        return "naive_outcome_rate"

    @property
    def family(self) -> str:
        return "naive_outcome_frequency"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def spec(self) -> dict[str, Any]:
        return self.export_config()

    @property
    def probabilities(self) -> tuple[float, float, float]:
        return self._probabilities

    def fit(self, played: pd.DataFrame) -> NaiveOutcomeRateBaseline:
        training = _played_with_results(played)
        counts = {"H": 0.0, "D": 0.0, "A": 0.0}
        for row in training.itertuples(index=False):
            counts[_outcome_from_goals(row.home_goals, row.away_goals)] += 1.0

        total = float(len(training)) + self.prior_strength
        home = (counts["H"] + self.prior_home * self.prior_strength) / total
        draw = (counts["D"] + self.prior_draw * self.prior_strength) / total
        away = (counts["A"] + self.prior_away * self.prior_strength) / total
        self._probabilities = _normalise_probabilities(home, draw, away)
        self._training_matches = int(len(training))
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        p_home, p_draw, p_away = self._probabilities
        rows = []
        for row in fixtures.itertuples(index=False):
            rows.append(
                {
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "match_date": _match_date(row),
                    "round": _round_label(row),
                    "p_home_win": p_home,
                    "p_draw": p_draw,
                    "p_away_win": p_away,
                    "model_name": self.name,
                    "model_family": self.family,
                    "model_version": self.version,
                    "training_matches": self._training_matches,
                }
            )
        return pd.DataFrame(rows)

    def export_config(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "model_family": self.family,
            "model_version": self.version,
            "required_columns": list(REQUIRED_COLUMNS),
            "config": {
                "prior_home": self.prior_home,
                "prior_draw": self.prior_draw,
                "prior_away": self.prior_away,
                "prior_strength": self.prior_strength,
            },
        }


@dataclass
class EloBaseline:
    """Simple match-result Elo baseline with fixed home advantage."""

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 65.0
    draw_base_probability: float = 0.26
    draw_rating_scale: float = 400.0

    ratings: dict[str, float] = field(default_factory=dict, init=False)
    _training_matches: int = field(default=0, init=False)

    @property
    def name(self) -> str:
        return "elo_baseline"

    @property
    def family(self) -> str:
        return "elo_result_rating"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def spec(self) -> dict[str, Any]:
        return self.export_config()

    def fit(self, played: pd.DataFrame) -> EloBaseline:
        training = _played_with_results(played).sort_values(
            ["match_date", "home_team", "away_team"],
            kind="mergesort",
        )
        for row in training.itertuples(index=False):
            self._update_ratings(
                str(row.home_team),
                str(row.away_team),
                int(row.home_goals),
                int(row.away_goals),
            )
        self._training_matches = int(len(training))
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in fixtures.itertuples(index=False):
            p_home, p_draw, p_away = self._fixture_probabilities(str(row.home_team), str(row.away_team))
            rows.append(
                {
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "match_date": _match_date(row),
                    "round": _round_label(row),
                    "p_home_win": p_home,
                    "p_draw": p_draw,
                    "p_away_win": p_away,
                    "model_name": self.name,
                    "model_family": self.family,
                    "model_version": self.version,
                    "training_matches": self._training_matches,
                    "home_rating": self._rating(str(row.home_team)),
                    "away_rating": self._rating(str(row.away_team)),
                }
            )
        return pd.DataFrame(rows)

    def export_config(self) -> dict[str, Any]:
        config = asdict(self)
        config.pop("ratings", None)
        config.pop("_training_matches", None)
        return {
            "model_name": self.name,
            "model_family": self.family,
            "model_version": self.version,
            "required_columns": list(REQUIRED_COLUMNS),
            "config": config,
        }

    def _update_ratings(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> None:
        home_rating = self._rating(home_team)
        away_rating = self._rating(away_team)
        expected_home = self._expected_home_score(home_rating, away_rating)
        actual_home = _elo_actual_score(home_goals, away_goals)
        change = self.k_factor * (actual_home - expected_home)
        self.ratings[home_team] = home_rating + change
        self.ratings[away_team] = away_rating - change

    def _fixture_probabilities(self, home_team: str, away_team: str) -> tuple[float, float, float]:
        home_rating = self._rating(home_team)
        away_rating = self._rating(away_team)
        expected_home = self._expected_home_score(home_rating, away_rating)
        rating_gap = abs((home_rating + self.home_advantage) - away_rating)
        p_draw = self.draw_base_probability * math.exp(-rating_gap / self.draw_rating_scale)
        p_draw = min(max(p_draw, 0.05), 0.45)
        non_draw = 1.0 - p_draw
        p_home = non_draw * expected_home
        p_away = non_draw * (1.0 - expected_home)
        return _normalise_probabilities(p_home, p_draw, p_away)

    def _expected_home_score(self, home_rating: float, away_rating: float) -> float:
        diff = home_rating + self.home_advantage - away_rating
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def _rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)


def _played_with_results(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required baseline columns: {missing}")
    played = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()
    if "match_date" in played.columns:
        played["match_date"] = pd.to_datetime(played["match_date"], format="ISO8601", errors="raise")
    return played


def _outcome_from_goals(home_goals: Any, away_goals: Any) -> str:
    home = int(home_goals)
    away = int(away_goals)
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"


def _elo_actual_score(home_goals: int, away_goals: int) -> float:
    outcome = _outcome_from_goals(home_goals, away_goals)
    if outcome == "H":
        return 1.0
    if outcome == "A":
        return 0.0
    return 0.5


def _normalise_probabilities(home: float, draw: float, away: float) -> tuple[float, float, float]:
    total = home + draw + away
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (home / total, draw / total, away / total)


def _match_date(row: Any) -> str:
    value = getattr(row, "match_date", None)
    if value is None or pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date())


def _round_label(row: Any) -> str:
    value = getattr(row, "round_label", "")
    if value is None or pd.isna(value):
        return ""
    return str(value)

