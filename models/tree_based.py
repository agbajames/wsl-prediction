"""
Conservative tree-based challenger models.

The random forest in this module is intentionally small and dependency-light.
It reuses leakage-safe feature generation from the improved logistic challenger
and implements shallow bagged classification trees with NumPy only.
"""

from __future__ import annotations

import math
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
class _TreeNode:
    probabilities: np.ndarray
    feature_index: int | None = None
    threshold: float | None = None
    left: _TreeNode | None = None
    right: _TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None or self.left is None or self.right is None


@dataclass
class RandomForestChallenger:
    """Small deterministic random-forest challenger for offline evaluation."""

    n_estimators: int = 50
    max_depth: int = 3
    min_samples_leaf: int = 5
    max_features: str | int = "sqrt"
    max_thresholds: int = 8
    bootstrap_fraction: float = 0.85
    class_prior_strength: float = 3.0
    min_training_matches: int = 12
    random_seed: int = 42
    feature_group: str = "xg"
    recent_window: int = 4

    _feature_builder: ImprovedTeamFormFeatureBuilder = field(init=False, repr=False)
    _trees: list[_TreeNode] = field(default_factory=list, init=False, repr=False)
    _feature_importances: np.ndarray | None = field(default=None, init=False, repr=False)
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
        return "random_forest"

    @property
    def family(self) -> str:
        return "conservative_random_forest"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def spec(self) -> dict[str, Any]:
        return self.export_config()

    @property
    def feature_importances(self) -> dict[str, float]:
        if self._feature_importances is None:
            return {}
        return {
            name: float(value)
            for name, value in zip(self._feature_builder.feature_names, self._feature_importances, strict=True)
        }

    def fit(self, played: pd.DataFrame) -> RandomForestChallenger:
        x_train = self._feature_builder.transform_training(played)
        outcomes = _training_outcomes(played)
        self._training_matches = len(outcomes)
        self._trees = []
        self._feature_importances = np.zeros(x_train.shape[1], dtype=float)
        self._fallback_model = None

        if self._should_use_fallback(outcomes):
            self._fit_fallback(played)
            return self

        y_train = np.asarray([OUTCOME_TO_INDEX[outcome] for outcome in outcomes], dtype=int)
        rng = np.random.default_rng(self.random_seed)
        sample_size = max(
            int(round(self.bootstrap_fraction * len(y_train))),
            min(len(y_train), self.min_samples_leaf * 2),
        )
        sample_size = min(sample_size, len(y_train))
        for _ in range(self.n_estimators):
            indices = rng.choice(len(y_train), size=sample_size, replace=True)
            tree = self._build_tree(
                x_train[indices],
                y_train[indices],
                depth=0,
                rng=rng,
                total_samples=len(y_train),
            )
            self._trees.append(tree)

        total_importance = float(self._feature_importances.sum())
        if total_importance > 0:
            self._feature_importances = self._feature_importances / total_importance
        self._feature_builder.fit(played)
        self._fit_mode = "internal_random_forest"
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        if self._fallback_model is not None:
            predictions = self._fallback_model.predict(fixtures)
            predictions["model_name"] = self.name
            predictions["model_family"] = self.family
            predictions["model_version"] = self.version
            predictions["fit_mode"] = self._fit_mode
            return predictions

        if not self._trees:
            raise RuntimeError("RandomForestChallenger must be fitted before predict().")

        x_test = self._feature_builder.transform_fixtures(fixtures)
        probabilities = np.zeros((len(x_test), len(OUTCOME_LABELS)), dtype=float)
        for tree in self._trees:
            probabilities += np.vstack([_predict_tree(tree, row) for row in x_test])
        probabilities /= len(self._trees)
        row_sums = probabilities.sum(axis=1)
        probabilities[row_sums == 0.0] = 1 / len(OUTCOME_LABELS)
        row_sums = probabilities.sum(axis=1)
        probabilities = probabilities / row_sums[:, None]

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
            "_trees",
            "_feature_importances",
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

    def _should_use_fallback(self, outcomes: list[str]) -> bool:
        return self._training_matches < self.min_training_matches or len(set(outcomes)) < 2

    def _fit_fallback(self, played: pd.DataFrame) -> None:
        self._fallback_model = NaiveOutcomeRateBaseline().fit(played)
        self._feature_builder.fit(played)
        self._fit_mode = "naive_small_sample_fallback"

    def _build_tree(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        *,
        depth: int,
        rng: np.random.Generator,
        total_samples: int,
    ) -> _TreeNode:
        probabilities = _leaf_probabilities(y_values, prior_strength=self.class_prior_strength)
        if depth >= self.max_depth or len(y_values) < self.min_samples_leaf * 2 or len(set(y_values.tolist())) <= 1:
            return _TreeNode(probabilities=probabilities)

        split = self._best_split(x_values, y_values, rng)
        if split is None:
            return _TreeNode(probabilities=probabilities)

        feature_index, threshold, gain, left_mask = split
        self._feature_importances[feature_index] += gain * (len(y_values) / max(total_samples, 1))
        return _TreeNode(
            probabilities=probabilities,
            feature_index=feature_index,
            threshold=threshold,
            left=self._build_tree(
                x_values[left_mask],
                y_values[left_mask],
                depth=depth + 1,
                rng=rng,
                total_samples=total_samples,
            ),
            right=self._build_tree(
                x_values[~left_mask],
                y_values[~left_mask],
                depth=depth + 1,
                rng=rng,
                total_samples=total_samples,
            ),
        )

    def _best_split(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, float, float, np.ndarray] | None:
        parent_impurity = _gini_impurity(y_values)
        best: tuple[int, float, float, np.ndarray] | None = None
        for feature_index in self._candidate_features(x_values.shape[1], rng):
            values = x_values[:, feature_index]
            for threshold in _candidate_thresholds(values, self.max_thresholds):
                left_mask = values <= threshold
                left_n = int(left_mask.sum())
                right_n = len(y_values) - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                gain = parent_impurity
                gain -= (left_n / len(y_values)) * _gini_impurity(y_values[left_mask])
                gain -= (right_n / len(y_values)) * _gini_impurity(y_values[~left_mask])
                if gain <= 1e-12:
                    continue
                if best is None or gain > best[2]:
                    best = (feature_index, float(threshold), float(gain), left_mask)
        return best

    def _candidate_features(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        if isinstance(self.max_features, int):
            count = max(1, min(self.max_features, n_features))
        elif self.max_features == "sqrt":
            count = max(1, int(math.sqrt(n_features)))
        elif self.max_features == "all":
            count = n_features
        else:
            raise ValueError("max_features must be an integer, 'sqrt', or 'all'.")
        return rng.choice(n_features, size=count, replace=False)


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
        raise ValueError(f"Missing required random-forest columns: {missing}")
    played = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()
    played["match_date"] = pd.to_datetime(played["match_date"], format="ISO8601", errors="raise")
    return played.sort_values(["match_date", "home_team", "away_team"], kind="mergesort")


def _candidate_thresholds(values: np.ndarray, max_thresholds: int) -> np.ndarray:
    unique = np.unique(values)
    if len(unique) <= 1:
        return np.asarray([], dtype=float)
    thresholds = (unique[:-1] + unique[1:]) / 2.0
    if len(thresholds) <= max_thresholds:
        return thresholds
    quantiles = np.linspace(0.1, 0.9, max_thresholds)
    return np.unique(np.quantile(thresholds, quantiles))


def _gini_impurity(y_values: np.ndarray) -> float:
    if len(y_values) == 0:
        return 0.0
    counts = np.bincount(y_values, minlength=len(OUTCOME_LABELS)).astype(float)
    probabilities = counts / counts.sum()
    return float(1.0 - np.sum(probabilities**2))


def _leaf_probabilities(y_values: np.ndarray, *, prior_strength: float) -> np.ndarray:
    counts = np.bincount(y_values, minlength=len(OUTCOME_LABELS)).astype(float)
    counts += prior_strength / len(OUTCOME_LABELS)
    return counts / counts.sum()


def _predict_tree(tree: _TreeNode, row: np.ndarray) -> np.ndarray:
    if tree.is_leaf:
        return tree.probabilities
    if row[tree.feature_index] <= tree.threshold:
        return _predict_tree(tree.left, row)
    return _predict_tree(tree.right, row)


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
