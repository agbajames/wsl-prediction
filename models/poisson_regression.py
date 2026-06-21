"""
Regularised Poisson regression challenger.

This is an offline evaluation model. It fits separate ridge-regularised
Poisson scoring-rate models for home and away goals inside each backtest fold,
then converts expected goals into home/draw/away probabilities with an
independent Poisson scoreline grid.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)

TARGET_COLUMNS = {
    "goals": ("home_goals", "away_goals"),
    "xg": ("home_xg", "away_xg"),
    "np_xg": ("home_np_xg", "away_np_xg"),
}


@dataclass
class PoissonRegressionChallenger:
    """Small-sample ridge Poisson regression challenger for WSL evaluation."""

    target_source: str = "goals"
    ridge_alpha: float = 1.0
    learning_rate: float = 0.01
    max_iter: int = 1000
    tolerance: float = 1e-6
    max_goals: int = 8
    min_rate: float = 0.05
    max_rate: float = 5.0

    _resolved_target_source: str = field(default="unfitted", init=False)
    _teams: list[str] = field(default_factory=list, init=False)
    _team_index: dict[str, int] = field(default_factory=dict, init=False)
    _home_weights: np.ndarray | None = field(default=None, init=False, repr=False)
    _away_weights: np.ndarray | None = field(default=None, init=False, repr=False)
    _training_matches: int = field(default=0, init=False)
    _home_iterations: int = field(default=0, init=False)
    _away_iterations: int = field(default=0, init=False)

    @property
    def name(self) -> str:
        return "poisson_regression"

    @property
    def family(self) -> str:
        return "regularised_poisson_regression"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def spec(self) -> dict[str, Any]:
        return self.export_config()

    def fit(self, played: pd.DataFrame) -> PoissonRegressionChallenger:
        """Fit leakage-safe home and away scoring-rate models from training rows."""
        training = _played_with_results(played)
        home_target, away_target, resolved_source = _resolve_target_columns(training, self.target_source)
        self._resolved_target_source = resolved_source
        self._training_matches = int(len(training))
        self._teams = sorted(set(training["home_team"].astype(str)) | set(training["away_team"].astype(str)))
        self._team_index = {team: idx for idx, team in enumerate(self._teams)}

        home_x = self._design_matrix(training, attacking_column="home_team", defending_column="away_team")
        away_x = self._design_matrix(training, attacking_column="away_team", defending_column="home_team")
        home_y = training[home_target].astype(float).to_numpy()
        away_y = training[away_target].astype(float).to_numpy()

        self._home_weights, self._home_iterations = _fit_poisson_ridge(
            home_x,
            home_y,
            ridge_alpha=self.ridge_alpha,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            min_rate=self.min_rate,
            max_rate=self.max_rate,
        )
        self._away_weights, self._away_iterations = _fit_poisson_ridge(
            away_x,
            away_y,
            ridge_alpha=self.ridge_alpha,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            min_rate=self.min_rate,
            max_rate=self.max_rate,
        )
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Predict fixture-level H/D/A probabilities."""
        if self._home_weights is None or self._away_weights is None:
            raise RuntimeError("PoissonRegressionChallenger must be fitted before predict().")

        rows = []
        for row in fixtures.itertuples(index=False):
            home_team = str(row.home_team)
            away_team = str(row.away_team)
            expected_home = self.expected_home_goals(home_team, away_team)
            expected_away = self.expected_away_goals(home_team, away_team)
            p_home, p_draw, p_away = win_draw_loss_probabilities(expected_home, expected_away, self.max_goals)
            rows.append(
                {
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "match_date": _match_date(row),
                    "round": _round_label(row),
                    "p_home_win": p_home,
                    "p_draw": p_draw,
                    "p_away_win": p_away,
                    "predicted_outcome": _predicted_outcome(p_home, p_draw, p_away),
                    "expected_home_goals": expected_home,
                    "expected_away_goals": expected_away,
                    "model_name": self.name,
                    "model_family": self.family,
                    "model_version": self.version,
                    "target_source": self._resolved_target_source,
                    "training_matches": self._training_matches,
                    "home_fit_iterations": self._home_iterations,
                    "away_fit_iterations": self._away_iterations,
                }
            )
        return pd.DataFrame(rows)

    def expected_home_goals(self, home_team: str, away_team: str) -> float:
        """Return capped expected home goals for a fixture."""
        if self._home_weights is None:
            raise RuntimeError("PoissonRegressionChallenger must be fitted before expected goals are available.")
        x_values = self._fixture_features(home_team, away_team)
        return _rate_from_weights(x_values, self._home_weights, min_rate=self.min_rate, max_rate=self.max_rate)

    def expected_away_goals(self, home_team: str, away_team: str) -> float:
        """Return capped expected away goals for a fixture."""
        if self._away_weights is None:
            raise RuntimeError("PoissonRegressionChallenger must be fitted before expected goals are available.")
        x_values = self._fixture_features(away_team, home_team)
        return _rate_from_weights(x_values, self._away_weights, min_rate=self.min_rate, max_rate=self.max_rate)

    def export_config(self) -> dict[str, Any]:
        config = asdict(self)
        for private_key in (
            "_resolved_target_source",
            "_teams",
            "_team_index",
            "_home_weights",
            "_away_weights",
            "_training_matches",
            "_home_iterations",
            "_away_iterations",
        ):
            config.pop(private_key, None)
        return {
            "model_name": self.name,
            "model_family": self.family,
            "model_version": self.version,
            "required_columns": list(REQUIRED_COLUMNS),
            "config": config,
            "resolved_target_source": self._resolved_target_source,
        }

    def _design_matrix(self, df: pd.DataFrame, *, attacking_column: str, defending_column: str) -> np.ndarray:
        rows = [
            self._fixture_features(str(row[attacking_column]), str(row[defending_column]))
            for _, row in df.iterrows()
        ]
        return np.vstack(rows)

    def _fixture_features(self, attacking_team: str, defending_team: str) -> np.ndarray:
        n_teams = len(self._teams)
        features = np.zeros(1 + 2 * n_teams, dtype=float)
        features[0] = 1.0
        attack_idx = self._team_index.get(attacking_team)
        defence_idx = self._team_index.get(defending_team)
        if attack_idx is not None:
            features[1 + attack_idx] = 1.0
        if defence_idx is not None:
            features[1 + n_teams + defence_idx] = 1.0
        return features


def win_draw_loss_probabilities(lambda_home: float, lambda_away: float, max_goals: int) -> tuple[float, float, float]:
    """Convert expected goals to H/D/A probabilities using a Poisson grid."""
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1.")
    home_pmf = [_poisson_pmf(score, lambda_home) for score in range(max_goals + 1)]
    away_pmf = [_poisson_pmf(score, lambda_away) for score in range(max_goals + 1)]
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for home_goals, home_prob in enumerate(home_pmf):
        for away_goals, away_prob in enumerate(away_pmf):
            probability = home_prob * away_prob
            if home_goals > away_goals:
                p_home += probability
            elif home_goals == away_goals:
                p_draw += probability
            else:
                p_away += probability
    return _normalise_probabilities(p_home, p_draw, p_away)


def _fit_poisson_ridge(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    ridge_alpha: float,
    learning_rate: float,
    max_iter: int,
    tolerance: float,
    min_rate: float,
    max_rate: float,
) -> tuple[np.ndarray, int]:
    if len(y_values) == 0:
        raise ValueError("Poisson regression requires at least one training row.")
    weights = np.zeros(x_values.shape[1], dtype=float)
    weights[0] = math.log(_clip(float(np.mean(y_values)), min_rate, max_rate))
    penalty = np.ones_like(weights)
    penalty[0] = 0.0
    previous_loss = math.inf

    for iteration in range(1, max_iter + 1):
        linear = np.clip(x_values @ weights, math.log(min_rate), math.log(max_rate))
        rates = np.exp(linear)
        residual = rates - y_values
        gradient = (x_values.T @ residual) / len(y_values)
        gradient += (ridge_alpha / len(y_values)) * penalty * weights
        weights -= learning_rate * gradient

        loss = float(np.mean(rates - y_values * linear))
        loss += float(0.5 * ridge_alpha * np.sum((penalty * weights) ** 2) / len(y_values))
        if abs(previous_loss - loss) < tolerance:
            return weights, iteration
        previous_loss = loss

    return weights, max_iter


def _played_with_results(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Poisson regression columns: {missing}")
    played = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()
    if played.empty:
        raise ValueError("Poisson regression model requires at least one played match.")
    played["match_date"] = pd.to_datetime(played["match_date"], format="ISO8601", errors="raise")
    return played


def _resolve_target_columns(df: pd.DataFrame, preferred_source: str) -> tuple[str, str, str]:
    sources = [preferred_source, "goals"]
    for source in sources:
        if source not in TARGET_COLUMNS:
            continue
        home_column, away_column = TARGET_COLUMNS[source]
        if {home_column, away_column}.issubset(df.columns) and df[[home_column, away_column]].notna().all().all():
            return home_column, away_column, source
    raise ValueError("No usable Poisson regression target found. Expected goals, xg, or np_xg columns.")


def _rate_from_weights(x_values: np.ndarray, weights: np.ndarray, *, min_rate: float, max_rate: float) -> float:
    linear = float(np.dot(x_values, weights))
    return _clip(math.exp(_clip(linear, math.log(min_rate), math.log(max_rate))), min_rate, max_rate)


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def _normalise_probabilities(home: float, draw: float, away: float) -> tuple[float, float, float]:
    total = home + draw + away
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (home / total, draw / total, away / total)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


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


def _predicted_outcome(p_home: float, p_draw: float, p_away: float) -> str:
    return ("H", "D", "A")[max(range(3), key=(p_home, p_draw, p_away).__getitem__)]
