"""
Calibration reporting helpers for probabilistic WSL predictions.

These functions build on ``evaluation.metrics`` and accept the same
home/draw/away probability ordering: H, D, A.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import calibration_bins, confidence_bucket_summary, outcome_indices, validate_probabilities


def build_calibration_bins(
    probabilities: list[list[float]] | np.ndarray,
    outcomes: list[str] | np.ndarray,
    *,
    n_bins: int = 5,
    min_bin_size: int = 1,
) -> list[dict[str, Any]]:
    """Return deterministic calibration bins with small-sample flags."""
    if min_bin_size < 1:
        raise ValueError("min_bin_size must be at least 1.")

    bins = calibration_bins(probabilities, outcomes, n_bins=n_bins)
    for row in bins:
        row["is_sparse"] = row["count"] < min_bin_size
        row["calibration_gap"] = (
            None
            if row["mean_confidence"] is None or row["observed_accuracy"] is None
            else round(row["mean_confidence"] - row["observed_accuracy"], 4)
        )
    return bins


def build_confidence_buckets(
    probabilities: list[list[float]] | np.ndarray,
    outcomes: list[str] | np.ndarray,
    *,
    min_bucket_size: int = 1,
) -> list[dict[str, Any]]:
    """Return low/medium/high confidence summaries with sparse-bucket flags."""
    if min_bucket_size < 1:
        raise ValueError("min_bucket_size must be at least 1.")

    buckets = confidence_bucket_summary(probabilities, outcomes)
    for row in buckets:
        row["is_sparse"] = row["count"] < min_bucket_size
        row["calibration_gap"] = (
            None
            if row["mean_confidence"] is None or row["accuracy"] is None
            else round(row["mean_confidence"] - row["accuracy"], 4)
        )
    return buckets


def reliability_table(
    probabilities: list[list[float]] | np.ndarray,
    outcomes: list[str] | np.ndarray,
    *,
    n_bins: int = 5,
    min_bin_size: int = 1,
) -> pd.DataFrame:
    """Return calibration bins as a DataFrame suitable for reports."""
    return pd.DataFrame(
        build_calibration_bins(
            probabilities,
            outcomes,
            n_bins=n_bins,
            min_bin_size=min_bin_size,
        )
    )


def calibration_summary(
    probabilities: list[list[float]] | np.ndarray,
    outcomes: list[str] | np.ndarray,
    *,
    n_bins: int = 5,
    min_bin_size: int = 1,
) -> dict[str, Any]:
    """Return calibration bins, confidence buckets, and aggregate confidence stats."""
    probs = validate_probabilities(probabilities)
    actual = outcome_indices(outcomes)
    if len(probs) != len(actual):
        raise ValueError("Probabilities and outcomes must have the same length.")

    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == actual).astype(float)
    return {
        "n_matches": int(len(probs)),
        "mean_confidence": round(float(confidence.mean()), 4) if len(confidence) else None,
        "accuracy": round(float(correct.mean()), 4) if len(correct) else None,
        "calibration_bins": build_calibration_bins(
            probs,
            actual,
            n_bins=n_bins,
            min_bin_size=min_bin_size,
        ),
        "confidence_buckets": build_confidence_buckets(
            probs,
            actual,
            min_bucket_size=min_bin_size,
        ),
    }

