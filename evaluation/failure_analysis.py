"""
Failure-analysis helpers for evaluation reports.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from evaluation.compare import extract_outcomes, extract_probabilities
from evaluation.metrics import OUTCOME_LABELS, outcome_indices, validate_probabilities


def scored_prediction_rows(rows: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    """Return rows with actual probability, log loss, favourite and correctness."""
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    probs = validate_probabilities(extract_probabilities(df))
    actual = outcome_indices(extract_outcomes(df))

    scored = df.reset_index(drop=True).copy()
    scored["actual_outcome"] = [OUTCOME_LABELS[idx] for idx in actual]
    scored["predicted_outcome"] = [OUTCOME_LABELS[idx] for idx in probs.argmax(axis=1)]
    scored["predicted_confidence"] = probs.max(axis=1)
    scored["actual_probability"] = probs[np.arange(len(probs)), actual]
    scored["row_log_loss"] = -np.log(np.clip(scored["actual_probability"].to_numpy(dtype=float), 1e-15, 1.0))
    scored["is_correct"] = scored["predicted_outcome"] == scored["actual_outcome"]
    scored["confidence_bucket"] = scored["predicted_confidence"].map(_confidence_bucket)
    return scored


def worst_misses(rows: pd.DataFrame | list[dict[str, Any]], *, n: int = 5) -> list[dict[str, Any]]:
    """Return the largest misses by row-level log loss."""
    return (
        scored_prediction_rows(rows)
        .sort_values(["row_log_loss", "actual_probability"], ascending=[False, True], kind="mergesort")
        .head(n)
        .to_dict(orient="records")
    )


def best_high_confidence_correct(
    rows: pd.DataFrame | list[dict[str, Any]],
    *,
    n: int = 5,
    min_confidence: float = 0.5,
) -> list[dict[str, Any]]:
    """Return correct predictions with the highest model confidence."""
    scored = scored_prediction_rows(rows)
    scored = scored[(scored["is_correct"]) & (scored["predicted_confidence"] >= min_confidence)]
    return (
        scored.sort_values(
            ["predicted_confidence", "actual_probability"],
            ascending=[False, False],
            kind="mergesort",
        )
        .head(n)
        .to_dict(orient="records")
    )


def favourite_breakdown(rows: pd.DataFrame | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarise correctness by predicted favourite and confidence bucket."""
    scored = scored_prediction_rows(rows)
    grouped = (
        scored.groupby(["predicted_outcome", "confidence_bucket"], sort=True)
        .agg(
            n=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
            mean_confidence=("predicted_confidence", "mean"),
            mean_actual_probability=("actual_probability", "mean"),
            mean_log_loss=("row_log_loss", "mean"),
        )
        .reset_index()
    )
    for column in ("accuracy", "mean_confidence", "mean_actual_probability", "mean_log_loss"):
        grouped[column] = grouped[column].round(4)
    return grouped.to_dict(orient="records")


def failure_analysis_summary(rows: pd.DataFrame | list[dict[str, Any]], *, n: int = 5) -> dict[str, Any]:
    """Return report-ready failure-analysis sections."""
    return {
        "worst_misses": worst_misses(rows, n=n),
        "best_high_confidence_correct": best_high_confidence_correct(rows, n=n),
        "favourite_breakdown": favourite_breakdown(rows),
    }


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.4:
        return "low"
    if confidence < 0.6:
        return "medium"
    return "high"

