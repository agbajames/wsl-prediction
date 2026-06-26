"""
Research-only tabular neural-network challenger.

This model is deliberately dependency-light. It reuses leakage-safe features
from the improved logistic challenger, fits feature scaling without looking at
future fixtures, and supports a small predeclared architecture ladder in NumPy.
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
    """Protocol-compatible research MLP for offline evaluation only."""

    hidden_units: int = 8
    hidden_layers: tuple[int, ...] | list[int] | None = None
    learning_rate: float = 0.03
    l2_penalty: float = 0.01
    dropout: float = 0.0
    max_iter: int = 500
    early_stopping: bool = True
    validation_fraction: float = 0.2
    early_stopping_patience: int = 20
    min_validation_matches: int = 3
    min_training_matches: int = 12
    random_seed: int = 42
    feature_group: str = "xg"
    recent_window: int = 4

    _feature_builder: ImprovedTeamFormFeatureBuilder = field(init=False, repr=False)
    _standard_mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _standard_scale: np.ndarray | None = field(default=None, init=False, repr=False)
    _weights: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _biases: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _fallback_model: NaiveOutcomeRateBaseline | None = field(default=None, init=False, repr=False)
    _training_matches: int = field(default=0, init=False)
    _fit_mode: str = field(default="unfitted", init=False)
    _training_history: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _fit_diagnostics: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._feature_builder = ImprovedTeamFormFeatureBuilder(
            feature_group=self.feature_group,
            recent_window=self.recent_window,
        )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in the range [0, 1).")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1.")
        if self.min_validation_matches < 1:
            raise ValueError("min_validation_matches must be at least 1.")

    @property
    def name(self) -> str:
        return "neural_network"

    @property
    def family(self) -> str:
        return "research_tabular_mlp"

    @property
    def version(self) -> str:
        return "v2"

    @property
    def spec(self) -> dict[str, Any]:
        return self.export_config()

    @property
    def scaler_mean(self) -> np.ndarray | None:
        return None if self._standard_mean is None else self._standard_mean.copy()

    @property
    def architecture(self) -> tuple[int, ...]:
        if self.hidden_layers is None:
            return (int(self.hidden_units),)
        return tuple(int(units) for units in self.hidden_layers)

    @property
    def training_history(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._training_history]

    @property
    def fit_diagnostics(self) -> dict[str, Any]:
        return dict(self._fit_diagnostics)

    @property
    def parameter_count(self) -> int:
        return int(sum(weight.size + bias.size for weight, bias in zip(self._weights, self._biases, strict=True)))

    def fit(self, played: pd.DataFrame) -> NeuralNetworkChallenger:
        x_all = self._feature_builder.transform_training(played)
        outcomes = _training_outcomes(played)
        self._training_matches = len(outcomes)
        self._fallback_model = None
        self._training_history = []
        self._fit_diagnostics = {}

        if self._should_use_fallback(outcomes):
            self._fit_fallback(played)
            return self

        y_all = np.asarray([OUTCOME_TO_INDEX[outcome] for outcome in outcomes], dtype=int)
        train_idx, validation_idx, early_stopping_skipped_reason = _time_ordered_validation_split(
            len(y_all),
            validation_fraction=self.validation_fraction,
            min_validation_matches=self.min_validation_matches,
            enabled=self.early_stopping,
        )
        x_fit = x_all[train_idx]
        y_fit = y_all[train_idx]
        self._standard_mean = x_fit.mean(axis=0)
        self._standard_scale = x_fit.std(axis=0)
        self._standard_scale[self._standard_scale == 0.0] = 1.0
        x_fit_scaled = self._standardize(x_fit)
        x_validation_scaled = self._standardize(x_all[validation_idx]) if len(validation_idx) else None
        y_validation = y_all[validation_idx] if len(validation_idx) else None

        rng = np.random.default_rng(self.random_seed)
        self._initialise_weights(x_fit_scaled.shape[1], rng)
        best_weights = [weight.copy() for weight in self._weights]
        best_biases = [bias.copy() for bias in self._biases]
        best_validation_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        early_stopping_triggered = False

        for epoch in range(1, self.max_iter + 1):
            probabilities, cache = self._forward(x_fit_scaled, training=True, rng=rng)
            self._backward(x_fit_scaled, y_fit, probabilities, cache)

            train_loss = _cross_entropy_loss(self._forward(x_fit_scaled)[0], y_fit)
            train_loss_with_l2 = train_loss + self._l2_loss()
            validation_loss = None
            validation_loss_with_l2 = None
            if x_validation_scaled is not None and y_validation is not None:
                validation_loss = _cross_entropy_loss(self._forward(x_validation_scaled)[0], y_validation)
                validation_loss_with_l2 = validation_loss + self._l2_loss()
                if validation_loss_with_l2 < best_validation_loss - 1e-12:
                    best_validation_loss = validation_loss_with_l2
                    best_epoch = epoch
                    best_weights = [weight.copy() for weight in self._weights]
                    best_biases = [bias.copy() for bias in self._biases]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.early_stopping_patience:
                        early_stopping_triggered = True
                        self._training_history.append(
                            _history_row(
                                epoch,
                                train_loss,
                                train_loss_with_l2,
                                validation_loss,
                                validation_loss_with_l2,
                            )
                        )
                        break
            else:
                best_epoch = epoch
                best_weights = [weight.copy() for weight in self._weights]
                best_biases = [bias.copy() for bias in self._biases]

            self._training_history.append(
                _history_row(epoch, train_loss, train_loss_with_l2, validation_loss, validation_loss_with_l2)
            )

        self._weights = best_weights
        self._biases = best_biases
        self._feature_builder.fit(played)
        self._fit_mode = "internal_softmax" if not self.architecture else "internal_mlp"
        self._fit_diagnostics = {
            "architecture": list(self.architecture),
            "parameter_count": self.parameter_count,
            "training_matches": self._training_matches,
            "fit_training_matches": int(len(train_idx)),
            "validation_matches": int(len(validation_idx)),
            "validation_fraction": self.validation_fraction,
            "early_stopping_enabled": bool(self.early_stopping),
            "early_stopping_skipped": early_stopping_skipped_reason is not None,
            "early_stopping_skipped_reason": early_stopping_skipped_reason,
            "early_stopping_triggered": early_stopping_triggered,
            "best_epoch": int(best_epoch),
            "epochs_run": int(len(self._training_history)),
            "best_validation_loss": None if best_validation_loss == float("inf") else float(best_validation_loss),
        }
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        if self._fallback_model is not None:
            predictions = self._fallback_model.predict(fixtures)
            predictions["model_name"] = self.name
            predictions["model_family"] = self.family
            predictions["model_version"] = self.version
            predictions["fit_mode"] = self._fit_mode
            return predictions

        if not self._weights or not self._biases:
            raise RuntimeError("NeuralNetworkChallenger must be fitted before predict().")

        x_test = self._feature_builder.transform_fixtures(fixtures)
        probabilities = self._forward(self._standardize(x_test))[0]

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
                    "architecture": _architecture_label(self.architecture),
                    "parameter_count": self.parameter_count,
                    "best_epoch": self._fit_diagnostics.get("best_epoch"),
                    "early_stopping_triggered": self._fit_diagnostics.get("early_stopping_triggered"),
                    "early_stopping_skipped": self._fit_diagnostics.get("early_stopping_skipped"),
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
            "_biases",
            "_fallback_model",
            "_training_matches",
            "_fit_mode",
            "_training_history",
            "_fit_diagnostics",
        ):
            config.pop(key, None)
        config["hidden_layers"] = list(self.architecture)
        return {
            "model_name": self.name,
            "model_family": self.family,
            "model_version": self.version,
            "required_columns": list(REQUIRED_COLUMNS),
            "features": list(self._feature_builder.feature_names),
            "config": config,
            "research_only": True,
        }

    def _initialise_weights(self, n_features: int, rng: np.random.Generator) -> None:
        layer_sizes = [n_features, *self.architecture, len(OUTCOME_LABELS)]
        if any(units <= 0 for units in layer_sizes):
            raise ValueError("All hidden layer sizes must be positive integers.")
        self._weights = []
        self._biases = []
        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:], strict=True):
            scale = float(np.sqrt(2.0 / max(input_size + output_size, 1)))
            self._weights.append(rng.normal(0.0, scale, size=(input_size, output_size)))
            self._biases.append(np.zeros(output_size, dtype=float))

    def _forward(
        self,
        x_values: np.ndarray,
        *,
        training: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        activations = x_values
        cache: list[dict[str, np.ndarray]] = []
        for layer_idx, (weights, bias) in enumerate(zip(self._weights, self._biases, strict=True)):
            layer_input = activations
            linear = layer_input @ weights + bias
            is_output = layer_idx == len(self._weights) - 1
            if is_output:
                activations = _softmax(linear)
                cache.append({"input": layer_input, "linear": linear})
                continue

            activated = np.tanh(linear)
            dropout_mask = np.ones_like(activated)
            if training and self.dropout > 0.0:
                if rng is None:
                    raise RuntimeError("Dropout training requires an RNG.")
                keep_probability = 1.0 - self.dropout
                dropout_mask = (rng.random(activated.shape) < keep_probability).astype(float) / keep_probability
                activated = activated * dropout_mask
            activations = activated
            cache.append(
                {"input": layer_input, "linear": linear, "activation": activated, "dropout_mask": dropout_mask}
            )
        return activations, cache

    def _backward(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        probabilities: np.ndarray,
        cache: list[dict[str, np.ndarray]],
    ) -> None:
        del x_values
        y_one_hot = np.zeros((len(y_values), len(OUTCOME_LABELS)), dtype=float)
        y_one_hot[np.arange(len(y_values)), y_values] = 1.0
        grad = (probabilities - y_one_hot) / len(y_values)
        grad_weights: list[np.ndarray] = []
        grad_biases: list[np.ndarray] = []

        for layer_idx in range(len(self._weights) - 1, -1, -1):
            layer_cache = cache[layer_idx]
            weights = self._weights[layer_idx]
            grad_weight = layer_cache["input"].T @ grad + self.l2_penalty * weights
            grad_bias = grad.sum(axis=0)
            grad_weights.append(grad_weight)
            grad_biases.append(grad_bias)

            if layer_idx > 0:
                previous_linear = cache[layer_idx - 1]["linear"]
                dropout_mask = cache[layer_idx - 1].get("dropout_mask", np.ones_like(previous_linear))
                grad = (grad @ weights.T) * dropout_mask * (1.0 - np.tanh(previous_linear) ** 2)

        for idx, (grad_weight, grad_bias) in enumerate(zip(reversed(grad_weights), reversed(grad_biases), strict=True)):
            self._weights[idx] -= self.learning_rate * grad_weight
            self._biases[idx] -= self.learning_rate * grad_bias

    def _l2_loss(self) -> float:
        return float(0.5 * self.l2_penalty * sum(float(np.sum(weight**2)) for weight in self._weights))

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
        self._fit_diagnostics = {
            "architecture": list(self.architecture),
            "parameter_count": 0,
            "training_matches": self._training_matches,
            "early_stopping_enabled": False,
            "early_stopping_skipped": True,
            "early_stopping_skipped_reason": "small_sample_or_single_class_fallback",
        }


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


def _time_ordered_validation_split(
    n_rows: int,
    *,
    validation_fraction: float,
    min_validation_matches: int,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    all_indices = np.arange(n_rows)
    if not enabled:
        return all_indices, np.asarray([], dtype=int), "early_stopping_disabled"
    validation_size = int(np.ceil(n_rows * validation_fraction))
    if validation_size < min_validation_matches:
        return all_indices, np.asarray([], dtype=int), "validation_split_too_small"
    if n_rows - validation_size < min_validation_matches:
        return all_indices, np.asarray([], dtype=int), "fit_split_too_small"
    return all_indices[:-validation_size], all_indices[-validation_size:], None


def _cross_entropy_loss(probabilities: np.ndarray, y_values: np.ndarray) -> float:
    actual_probability = probabilities[np.arange(len(y_values)), y_values]
    return float(-np.mean(np.log(np.clip(actual_probability, 1e-15, 1.0))))


def _history_row(
    epoch: int,
    train_loss: float,
    train_loss_with_l2: float,
    validation_loss: float | None,
    validation_loss_with_l2: float | None,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "train_loss_with_l2": float(train_loss_with_l2),
        "validation_loss": None if validation_loss is None else float(validation_loss),
        "validation_loss_with_l2": None if validation_loss_with_l2 is None else float(validation_loss_with_l2),
    }


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


def _architecture_label(architecture: tuple[int, ...]) -> str:
    return "[]" if not architecture else "[" + ",".join(str(units) for units in architecture) + "]"
