"""
evaluation/metrics.py
---------------------
Reusable evaluation metrics for three-way football predictions.

Outcome order is always:
    H = home win, D = draw, A = away win
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

OUTCOME_LABELS = ("H", "D", "A")
OUTCOME_TO_INDEX = {label: idx for idx, label in enumerate(OUTCOME_LABELS)}


def outcome_indices(outcomes: Sequence[str] | np.ndarray) -> np.ndarray:
    """Convert outcome labels or one-hot rows into class indices."""
    arr = np.asarray(outcomes)
    if arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError("One-hot outcomes must have exactly three columns.")
        return np.argmax(arr, axis=1).astype(int)

    indices: list[int] = []
    for outcome in arr.tolist():
        if isinstance(outcome, int) and outcome in (0, 1, 2):
            indices.append(outcome)
            continue
        label = str(outcome).upper()
        if label not in OUTCOME_TO_INDEX:
            raise ValueError(f"Invalid outcome label: {outcome!r}. Expected one of H, D, A.")
        indices.append(OUTCOME_TO_INDEX[label])
    return np.asarray(indices, dtype=int)


def one_hot_outcomes(outcomes: Sequence[str] | np.ndarray) -> np.ndarray:
    """Convert outcome labels or one-hot rows into a validated one-hot matrix."""
    arr = np.asarray(outcomes)
    if arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError("One-hot outcomes must have exactly three columns.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Outcomes must be finite.")
        if not np.all((arr == 0.0) | (arr == 1.0)):
            raise ValueError("One-hot outcomes must contain only 0 and 1.")
        if not np.allclose(arr.sum(axis=1), 1.0):
            raise ValueError("Each one-hot outcome row must sum to 1.")
        return arr.astype(float)

    indices = outcome_indices(outcomes)
    encoded = np.zeros((len(indices), 3), dtype=float)
    encoded[np.arange(len(indices)), indices] = 1.0
    return encoded


def validate_probabilities(probabilities: Iterable[Iterable[float]], normalize: bool = True) -> np.ndarray:
    """Validate 3-way probability rows and optionally normalize row sums.

    Raises:
        ValueError: if probabilities are not finite, contain negatives, have the
        wrong shape, or contain a row with zero total mass.
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("Probabilities must be a 2D array with exactly three columns.")
    if len(probs) == 0:
        raise ValueError("At least one probability row is required.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("Probabilities must be finite.")
    if np.any(probs < 0):
        raise ValueError("Probabilities must be non-negative.")

    row_sums = probs.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("Each probability row must have positive total mass.")

    if normalize:
        probs = probs / row_sums[:, None]
    elif not np.allclose(row_sums, 1.0):
        raise ValueError("Each probability row must sum to 1.")

    return probs


def _validate_lengths(probabilities: np.ndarray, outcomes: np.ndarray) -> None:
    if len(probabilities) != len(outcomes):
        raise ValueError("Probabilities and outcomes must have the same length.")


def brier_score_3way(probabilities: Iterable[Iterable[float]], outcomes: Sequence[str] | np.ndarray) -> float:
    """Mean multiclass Brier score for home/draw/away probabilities."""
    probs = validate_probabilities(probabilities)
    actual = one_hot_outcomes(outcomes)
    _validate_lengths(probs, actual)
    return float(np.mean(np.sum((probs - actual) ** 2, axis=1)))


def multiclass_log_loss(
    probabilities: Iterable[Iterable[float]],
    outcomes: Sequence[str] | np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Mean multiclass log loss for home/draw/away probabilities."""
    probs = validate_probabilities(probabilities)
    actual = one_hot_outcomes(outcomes)
    _validate_lengths(probs, actual)
    clipped = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(np.sum(actual * np.log(clipped), axis=1)))


def outcome_accuracy(probabilities: Iterable[Iterable[float]], outcomes: Sequence[str] | np.ndarray) -> float:
    """Share of matches where the highest-probability class matched the outcome."""
    probs = validate_probabilities(probabilities)
    actual = outcome_indices(outcomes)
    _validate_lengths(probs, actual)
    return float(np.mean(np.argmax(probs, axis=1) == actual))


def calibration_bins(
    probabilities: Iterable[Iterable[float]],
    outcomes: Sequence[str] | np.ndarray,
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    """Confidence calibration bins using max predicted probability."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    probs = validate_probabilities(probabilities)
    actual = outcome_indices(outcomes)
    _validate_lengths(probs, actual)

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == actual).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins: list[dict[str, Any]] = []
    for idx in range(n_bins):
        lower = float(edges[idx])
        upper = float(edges[idx + 1])
        mask = (confidence >= lower) & (confidence < upper)
        if idx == n_bins - 1:
            mask = mask | (confidence == upper)

        count = int(mask.sum())
        bins.append(
            {
                "bin": f"{lower:.0%}-{upper:.0%}",
                "lower": round(lower, 3),
                "upper": round(upper, 3),
                "count": count,
                "mean_confidence": round(float(confidence[mask].mean()), 4) if count else None,
                "observed_accuracy": round(float(correct[mask].mean()), 4) if count else None,
            }
        )
    return bins


def confidence_bucket_summary(
    probabilities: Iterable[Iterable[float]],
    outcomes: Sequence[str] | np.ndarray,
) -> list[dict[str, Any]]:
    """Simple low/medium/high confidence summary."""
    probs = validate_probabilities(probabilities)
    actual = outcome_indices(outcomes)
    _validate_lengths(probs, actual)

    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == actual).astype(float)
    buckets = [
        ("low", 0.0, 0.4),
        ("medium", 0.4, 0.6),
        ("high", 0.6, 1.0),
    ]

    summary: list[dict[str, Any]] = []
    for name, lower, upper in buckets:
        mask = (confidence >= lower) & (confidence < upper)
        if name == "high":
            mask = mask | (confidence == upper)
        count = int(mask.sum())
        summary.append(
            {
                "bucket": name,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": round(float(confidence[mask].mean()), 4) if count else None,
                "accuracy": round(float(correct[mask].mean()), 4) if count else None,
            }
        )
    return summary


def evaluate_prediction_set(
    probabilities: Iterable[Iterable[float]],
    outcomes: Sequence[str] | np.ndarray,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Return the standard evaluation metric bundle for a prediction set."""
    probs = validate_probabilities(probabilities)
    actual = outcome_indices(outcomes)
    _validate_lengths(probs, actual)
    return {
        "n_matches": int(len(probs)),
        "brier_score": brier_score_3way(probs, actual),
        "log_loss": multiclass_log_loss(probs, actual),
        "accuracy": outcome_accuracy(probs, actual),
        "calibration_bins": calibration_bins(probs, actual, n_bins=n_bins),
        "confidence_buckets": confidence_bucket_summary(probs, actual),
    }
