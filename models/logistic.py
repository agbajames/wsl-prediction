"""
Multinomial logistic regression challenger model.

The model uses leakage-aware team-form features derived from the training fold.
It uses scikit-learn when available; otherwise the default ``auto`` solver falls
back to a small internal softmax solver so local tests do not require live
services or extra dependencies. Setting ``solver="sklearn"`` without
scikit-learn raises a clear ImportError.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from features.team_form import (
    FEATURE_NAMES,
    ImprovedTeamFormFeatureBuilder,
    TeamFormFeatureBuilder,
    training_outcomes,
)
from models.baselines import NaiveOutcomeRateBaseline

OUTCOME_LABELS = ("H", "D", "A")
OUTCOME_TO_INDEX = {label: idx for idx, label in enumerate(OUTCOME_LABELS)}
REQUIRED_COLUMNS = (
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)


@dataclass
class LogisticRegressionChallenger:
    """Protocol-compatible multinomial logistic regression challenger."""

    min_training_matches: int = 6
    regularization_c: float = 1.0
    max_iter: int = 500
    learning_rate: float = 0.15
    l2_penalty: float = 0.1
    solver: str = "auto"

    _feature_builder: TeamFormFeatureBuilder = field(default_factory=TeamFormFeatureBuilder, init=False, repr=False)
    _standard_mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _standard_scale: np.ndarray | None = field(default=None, init=False, repr=False)
    _weights: np.ndarray | None = field(default=None, init=False, repr=False)
    _sklearn_model: Any | None = field(default=None, init=False, repr=False)
    _fallback_model: NaiveOutcomeRateBaseline | None = field(default=None, init=False, repr=False)
    _training_matches: int = field(default=0, init=False)
    _fit_mode: str = field(default="unfitted", init=False)

    @property
    def name(self) -> str:
        return "logistic_regression"

    @property
    def family(self) -> str:
        return "multinomial_logistic_regression"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def spec(self) -> dict[str, Any]:
        return self.export_config()

    @property
    def fit_mode(self) -> str:
        return self._fit_mode

    def fit(self, played: pd.DataFrame) -> LogisticRegressionChallenger:
        x_train = self._feature_builder.transform_training(played)
        outcomes = training_outcomes(played)
        self._training_matches = len(outcomes)

        if self._should_use_fallback(outcomes):
            self._fit_fallback(played)
            return self

        y_train = np.asarray([OUTCOME_TO_INDEX[outcome] for outcome in outcomes], dtype=int)
        self._standard_mean = x_train.mean(axis=0)
        self._standard_scale = x_train.std(axis=0)
        self._standard_scale[self._standard_scale == 0.0] = 1.0
        x_scaled = self._standardize(x_train)

        if self._should_use_sklearn():
            self._fit_sklearn(x_scaled, y_train)
        else:
            self._weights = _fit_softmax_regression(
                x_scaled,
                y_train,
                n_classes=len(OUTCOME_LABELS),
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                l2_penalty=self.l2_penalty,
            )
            self._fit_mode = "internal_softmax"

        self._feature_builder.fit(played)
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        if self._fallback_model is not None:
            predictions = self._fallback_model.predict(fixtures)
            predictions["model_name"] = self.name
            predictions["model_family"] = self.family
            predictions["model_version"] = self.version
            predictions["fit_mode"] = self._fit_mode
            return predictions

        if self._standard_mean is None or self._standard_scale is None:
            raise RuntimeError("LogisticRegressionChallenger must be fitted before predict().")

        x_test = self._feature_builder.transform_fixtures(fixtures)
        x_scaled = self._standardize(x_test)
        if self._sklearn_model is not None:
            probabilities = self._sklearn_model.predict_proba(x_scaled)
            probabilities = _align_sklearn_probabilities(probabilities, self._sklearn_model.classes_)
        elif self._weights is not None:
            probabilities = _softmax(_add_intercept(x_scaled) @ self._weights)
        else:
            raise RuntimeError("LogisticRegressionChallenger has no fitted classifier.")

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
            "_weights",
            "_sklearn_model",
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
            "features": list(FEATURE_NAMES),
            "config": config,
        }

    def _should_use_fallback(self, outcomes: list[str]) -> bool:
        return self._training_matches < self.min_training_matches or len(set(outcomes)) < 2

    def _fit_fallback(self, played: pd.DataFrame) -> None:
        self._fallback_model = NaiveOutcomeRateBaseline().fit(played)
        self._feature_builder.fit(played)
        self._fit_mode = "naive_small_sample_fallback"

    def _should_use_sklearn(self) -> bool:
        if self.solver not in {"auto", "sklearn", "internal"}:
            raise ValueError("solver must be one of: auto, sklearn, internal.")
        sklearn_available = _sklearn_logistic_regression() is not None
        if self.solver == "sklearn" and not sklearn_available:
            raise ImportError(
                "scikit-learn is required for solver='sklearn'. Install scikit-learn or use solver='auto'/'internal'."
            )
        return self.solver != "internal" and sklearn_available

    def _fit_sklearn(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        logistic_regression = _sklearn_logistic_regression()
        if logistic_regression is None:
            raise ImportError("scikit-learn is unavailable; use solver='auto' or solver='internal'.")
        self._sklearn_model = logistic_regression(
            C=self.regularization_c,
            max_iter=self.max_iter,
            multi_class="auto",
        )
        self._sklearn_model.fit(x_train, y_train)
        self._fit_mode = "sklearn"

    def _standardize(self, x_values: np.ndarray) -> np.ndarray:
        if self._standard_mean is None or self._standard_scale is None:
            raise RuntimeError("Feature scaler is not fitted.")
        return (x_values - self._standard_mean) / self._standard_scale


@dataclass
class ImprovedLogisticRegressionChallenger(LogisticRegressionChallenger):
    """Conservative richer-feature logistic challenger for Phase 8B."""

    min_training_matches: int = 10
    regularization_c: float = 0.5
    max_iter: int = 800
    learning_rate: float = 0.08
    l2_penalty: float = 0.2
    solver: str = "auto"
    feature_group: str = "xg"
    recent_window: int = 4

    _feature_builder: ImprovedTeamFormFeatureBuilder = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._feature_builder = ImprovedTeamFormFeatureBuilder(
            feature_group=self.feature_group,
            recent_window=self.recent_window,
        )

    @property
    def name(self) -> str:
        return "improved_logistic_regression"

    @property
    def family(self) -> str:
        return "multinomial_logistic_regression_rich_form"

    @property
    def version(self) -> str:
        return "v1"

    def export_config(self) -> dict[str, Any]:
        config = asdict(self)
        for key in (
            "_feature_builder",
            "_standard_mean",
            "_standard_scale",
            "_weights",
            "_sklearn_model",
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
        }


def _fit_softmax_regression(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    n_classes: int,
    max_iter: int,
    learning_rate: float,
    l2_penalty: float,
) -> np.ndarray:
    x_design = _add_intercept(x_values)
    weights = np.zeros((x_design.shape[1], n_classes), dtype=float)
    y_one_hot = np.zeros((len(y_values), n_classes), dtype=float)
    y_one_hot[np.arange(len(y_values)), y_values] = 1.0

    for _ in range(max_iter):
        probabilities = _softmax(x_design @ weights)
        gradient = (x_design.T @ (probabilities - y_one_hot)) / len(y_values)
        gradient[1:] += l2_penalty * weights[1:]
        weights -= learning_rate * gradient

    return weights


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _add_intercept(x_values: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x_values)), x_values])


def _align_sklearn_probabilities(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(OUTCOME_LABELS)), dtype=float)
    for column_idx, class_idx in enumerate(classes.astype(int)):
        aligned[:, class_idx] = probabilities[:, column_idx]
    row_sums = aligned.sum(axis=1)
    aligned[row_sums == 0.0] = 1 / len(OUTCOME_LABELS)
    row_sums = aligned.sum(axis=1)
    return aligned / row_sums[:, None]


def _sklearn_logistic_regression() -> Any | None:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    return LogisticRegression


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

