"""
Research-only tabular neural-network challenger.

This model is deliberately tiny and dependency-light. It reuses leakage-safe
features from the improved logistic challenger, fits feature scaling on each
training fold only, and trains a single-hidden-layer MLP with NumPy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from features.team_form import ImprovedTeamFormFeatureBuilder
from models.baselines import NaiveOutcomeRateBaseline
from models.logistic import OUTCOME_LABELS, OUTCOME_TO_INDEX

REQUIRED_COLUMNS = (
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)


@dataclass
class NeuralNetworkChallenger:
    """Tiny MLP proof-of-concept for offline evaluation only."""

    hidden_units: int = 8
    learning_rate: float = 0.03
    l2_penalty: float = 0.01
    max_iter: int = 500
    min_training_matches: int = 12
    random_seed: int = 42
    feature_group: str = "xg"
    recent_window: int = 4

    _feature_builder: ImprovedTeamFormFeatureBuilder = field(init=False, repr=False)
    _standard_mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _standard_scale: np.ndarray | None = field(default=None, init=False, repr=False)
    _weights_hidden: np.ndarray | None = field(default=None, init=False, repr=False)
    _bias_hidden: np.ndarray | None = field(default=None, init=False, repr=False)
    _weights_output: np.ndarray | None = field(default=None, init=False, repr=False)
    _bias_output: np.ndarray | None = field(default=None, init=False, repr=False)
    _fallback_model: NaiveOutcomeRateBaseline | None = field(default=None, init=False, repr=False)
    _training_matches: int = field(default=0, init=False)
    _fit_mode: str = field(default="unfitted", init=False)

    def __post_init__(self) -> None:
        self._feature_builder = ImprovedTeamFormFeatureBuilder(
            feature_group=self.feature_group,
            recent_window=self.recent_window,
        )

    @property
    def name(self) -> str:
        return "neural_network"

    @property
    def family(self) -> str:
        return "research_tiny_tabular_mlp"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def spec(self) -> dict[str, Any]:
        return self.export_config()

    @property
    def scaler_mean(self) -> np.ndarray | None:
        return None if self._standard_mean is None else self._standard_mean.copy()

    def fit(self, played: pd.DataFrame) -> NeuralNetworkChallenger:
        x_train = self._feature_builder.transform_training(played)
        outcomes = _training_outcomes(played)
        self._training_matches = len(outcomes)
        self._fallback_model = None

        if self._should_use_fallback(outcomes):
            self._fit_fallback(played)
            return self

        y_train = np.asarray([OUTCOME_TO_INDEX[outcome] for outcome in outcomes], dtype=int)
        y_one_hot = np.zeros((len(y_train), len(OUTCOME_LABELS)), dtype=float)
        y_one_hot[np.arange(len(y_train)), y_train] = 1.0
        self._standard_mean = x_train.mean(axis=0)
        self._standard_scale = x_train.std(axis=0)
        self._standard_scale[self._standard_scale == 0.0] = 1.0
        x_scaled = self._standardize(x_train)

        rng = np.random.default_rng(self.random_seed)
        n_features = x_scaled.shape[1]
        self._weights_hidden = rng.normal(0.0, 0.05, size=(n_features, self.hidden_units))
        self._bias_hidden = np.zeros(self.hidden_units, dtype=float)
        self._weights_output = rng.normal(0.0, 0.05, size=(self.hidden_units, len(OUTCOME_LABELS)))
        self._bias_output = np.zeros(len(OUTCOME_LABELS), dtype=float)

        for _ in range(self.max_iter):
            hidden_linear = x_scaled @ self._weights_hidden + self._bias_hidden
            hidden = np.tanh(hidden_linear)
            probabilities = _softmax(hidden @ self._weights_output + self._bias_output)

            output_error = (probabilities - y_one_hot) / len(y_train)
            grad_weights_output = hidden.T @ output_error + self.l2_penalty * self._weights_output
            grad_bias_output = output_error.sum(axis=0)
            hidden_error = (output_error @ self._weights_output.T) * (1.0 - hidden**2)
            grad_weights_hidden = x_scaled.T @ hidden_error + self.l2_penalty * self._weights_hidden
            grad_bias_hidden = hidden_error.sum(axis=0)

            self._weights_output -= self.learning_rate * grad_weights_output
            self._bias_output -= self.learning_rate * grad_bias_output
            self._weights_hidden -= self.learning_rate * grad_weights_hidden
            self._bias_hidden -= self.learning_rate * grad_bias_hidden

        self._feature_builder.fit(played)
        self._fit_mode = "internal_tiny_mlp"
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        if self._fallback_model is not None:
            predictions = self._fallback_model.predict(fixtures)
            predictions["model_name"] = self.name
            predictions["model_family"] = self.family
            predictions["model_version"] = self.version
            predictions["fit_mode"] = self._fit_mode
            return predictions

        if (
            self._weights_hidden is None
            or self._bias_hidden is None
            or self._weights_output is None
            or self._bias_output is None
        ):
            raise RuntimeError("NeuralNetworkChallenger must be fitted before predict().")

        x_test = self._feature_builder.transform_fixtures(fixtures)
        x_scaled = self._standardize(x_test)
        probabilities = _softmax(np.tanh(x_scaled @ self._weights_hidden + self._bias_hidden) @ self._weights_output + self._bias_output)

        rows = []
        for idx, row in enumerate(fixtures.itertuples(index=False)):
            p_home, p_draw, p_away = probabilities[idx]
            rows.append(
                {
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "match_date": _match_date(row),
                    "round": _round_label(row),
                    "p_home_win": float(p_home),
                    "p_draw": float(p_draw),
                    "p_away_win": float(p_away),
                    "predicted_outcome": _predicted_outcome(float(p_home), float(p_draw), float(p_away)),
                    "model_name": self.name,
                    "model_family": self.family,
                    "model_version": self.version,
                    "fit_mode": self._fit_mode,
                    "training_matches": self._training_matches,
                }
            )
        return pd.DataFrame(rows)

    def export_config(self) -> dict[str, Any]:
        config = asdict(self)
        for key in (
            "_feature_builder",
            "_standard_mean",
            "_standard_scale",
            "_weights_hidden",
            "_bias_hidden",
            "_weights_output",
            "_bias_output",
            "_fallback_model",
            "_training_matches",
            "_fit_mode",
        ):
            config.pop(key, None)
        return {
            "model_name": self.name,
            "model_family": self.family,
            "model_version": self.version,
            "required_columns": list(REQUIRED_COLUMNS),
            "features": list(self._feature_builder.feature_names),
            "config": config,
            "research_only": True,
        }

    def _standardize(self, x_values: np.ndarray) -> np.ndarray:
        if self._standard_mean is None or self._standard_scale is None:
            raise RuntimeError("Feature scaler is not fitted.")
        return (x_values - self._standard_mean) / self._standard_scale

    def _should_use_fallback(self, outcomes: list[str]) -> bool:
        return self._training_matches < self.min_training_matches or len(set(outcomes)) < 2

    def _fit_fallback(self, played: pd.DataFrame) -> None:
        self._fallback_model = NaiveOutcomeRateBaseline().fit(played)
        self._feature_builder.fit(played)
        self._fit_mode = "naive_small_sample_fallback"


def _training_outcomes(played: pd.DataFrame) -> list[str]:
    training = _ordered_played_rows(played)
    outcomes = []
    for row in training.itertuples(index=False):
        if int(row.home_goals) > int(row.away_goals):
            outcomes.append("H")
        elif int(row.home_goals) < int(row.away_goals):
            outcomes.append("A")
        else:
            outcomes.append("D")
    return outcomes


def _ordered_played_rows(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required neural-network columns: {missing}")
    played = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()
    played["match_date"] = pd.to_datetime(played["match_date"], format="ISO8601", errors="raise")
    return played.sort_values(["match_date", "home_team", "away_team"], kind="mergesort")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


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
