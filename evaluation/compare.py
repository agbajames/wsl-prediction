"""
Model comparison helpers for evaluation reports.

Inputs are prediction-vs-result rows with home/draw/away probabilities and an
actual outcome column. The helpers work for a champion-only result set and for
future champion-vs-challenger comparisons.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from evaluation.metrics import brier_score_3way, multiclass_log_loss, outcome_accuracy

PROBABILITY_COLUMNS = ("p_home_win", "p_draw", "p_away_win")
MODEL_COLUMN = "model_name"
OUTCOME_CANDIDATES = ("actual_outcome", "outcome", "actual_result")


def summarize_model_results(
    results: pd.DataFrame | list[dict[str, Any]],
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Summarise probability-quality metrics for one model result set."""
    df = _coerce_dataframe(results)
    probabilities = extract_probabilities(df)
    outcomes = extract_outcomes(df)
    resolved_model = model_name or _single_or_default(df, MODEL_COLUMN, "model")
    return {
        "model_name": resolved_model,
        "n_matches": int(len(df)),
        "brier_score": brier_score_3way(probabilities, outcomes),
        "log_loss": multiclass_log_loss(probabilities, outcomes),
        "accuracy": outcome_accuracy(probabilities, outcomes),
    }


def compare_model_results(
    result_sets: pd.DataFrame | list[pd.DataFrame] | dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return ranked model comparison metrics as a DataFrame."""
    summaries: list[dict[str, Any]] = []

    if isinstance(result_sets, dict):
        for model_name, rows in result_sets.items():
            summaries.append(summarize_model_results(rows, model_name=model_name))
    elif isinstance(result_sets, list):
        for rows in result_sets:
            summaries.append(summarize_model_results(rows))
    else:
        df = _coerce_dataframe(result_sets)
        if MODEL_COLUMN in df.columns:
            for model_name, rows in df.groupby(MODEL_COLUMN, sort=True):
                summaries.append(summarize_model_results(rows, model_name=str(model_name)))
        else:
            summaries.append(summarize_model_results(df))

    comparison = pd.DataFrame(summaries)
    if comparison.empty:
        return comparison
    comparison = comparison.sort_values(
        ["log_loss", "brier_score", "accuracy", "model_name"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    comparison["rank"] = comparison.index + 1
    return comparison


def comparison_to_records(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    """Return report-ready records from a comparison DataFrame."""
    return comparison.to_dict(orient="records")


def extract_probabilities(df: pd.DataFrame) -> list[list[float]]:
    """Extract H/D/A probabilities from a result DataFrame."""
    missing = [column for column in PROBABILITY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    return df.loc[:, PROBABILITY_COLUMNS].astype(float).values.tolist()


def extract_outcomes(df: pd.DataFrame) -> list[str]:
    """Extract actual outcomes from common report/backtest column shapes."""
    for column in OUTCOME_CANDIDATES:
        if column in df.columns:
            return df[column].map(_normalise_outcome).tolist()

    if {"home_goals", "away_goals"}.issubset(df.columns):
        outcomes = []
        for home_goals, away_goals in df.loc[:, ["home_goals", "away_goals"]].itertuples(index=False):
            outcomes.append(_outcome_from_goals(home_goals, away_goals))
        return outcomes

    raise ValueError(
        "Missing actual outcome. Expected one of "
        f"{list(OUTCOME_CANDIDATES)} or home_goals/away_goals."
    )


def _coerce_dataframe(results: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    df = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    if df.empty:
        raise ValueError("At least one prediction-result row is required.")
    return df


def _single_or_default(df: pd.DataFrame, column: str, default: str) -> str:
    if column not in df.columns:
        return default
    values = sorted(df[column].dropna().astype(str).unique().tolist())
    return values[0] if len(values) == 1 else default


def _normalise_outcome(value: Any) -> str:
    label = str(value).strip().upper()
    aliases = {
        "HOME": "H",
        "HOME_WIN": "H",
        "H": "H",
        "DRAW": "D",
        "D": "D",
        "AWAY": "A",
        "AWAY_WIN": "A",
        "A": "A",
    }
    if label not in aliases:
        raise ValueError(f"Invalid actual outcome: {value!r}. Expected H, D, or A.")
    return aliases[label]


def _outcome_from_goals(home_goals: Any, away_goals: Any) -> str:
    home = int(home_goals)
    away = int(away_goals)
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"

