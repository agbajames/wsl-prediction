"""
Offline probability calibrators for champion evaluation experiments.

These helpers transform already-generated probability rows. They do not train
or alter the underlying champion model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.calibration import calibration_summary
from evaluation.compare import PROBABILITY_COLUMNS, extract_outcomes
from evaluation.failure_analysis import scored_prediction_rows
from evaluation.metrics import (
    brier_score_3way,
    multiclass_log_loss,
    one_hot_outcomes,
    outcome_accuracy,
    outcome_indices,
    validate_probabilities,
)

CHAMPION_MODEL = "champion_dc_xg"


@dataclass(frozen=True)
class TemperatureShrinkageCalibrator:
    """Temperature-scale probabilities, then shrink them toward base rates."""

    temperature: float
    shrinkage: float
    base_rates: tuple[float, float, float]

    def transform(self, probabilities: list[list[float]] | np.ndarray) -> np.ndarray:
        """Return calibrated H/D/A probabilities."""
        scaled = temperature_scale_probabilities(probabilities, self.temperature)
        base = np.asarray(self.base_rates, dtype=float)
        calibrated = ((1.0 - self.shrinkage) * scaled) + (self.shrinkage * base)
        return validate_probabilities(calibrated)

    def to_dict(self) -> dict[str, Any]:
        """Return report-ready calibration parameters."""
        return {
            "method": "temperature_scaling_with_base_rate_shrinkage",
            "temperature": round(float(self.temperature), 4),
            "shrinkage": round(float(self.shrinkage), 4),
            "base_rates": {
                "home_win": round(float(self.base_rates[0]), 4),
                "draw": round(float(self.base_rates[1]), 4),
                "away_win": round(float(self.base_rates[2]), 4),
            },
        }


def load_model_comparison_payload(path: Path) -> dict[str, Any]:
    """Load a local model-comparison JSON artefact."""
    if not path.exists():
        raise FileNotFoundError(f"Model comparison JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model comparison JSON must contain an object payload.")
    if "prediction_rows" not in payload:
        raise ValueError("Model comparison JSON is missing 'prediction_rows'.")
    return payload


def extract_champion_predictions(
    payload: dict[str, Any],
    *,
    champion_model: str = CHAMPION_MODEL,
) -> pd.DataFrame:
    """Return champion prediction rows with probability columns normalized to 0-1."""
    rows = pd.DataFrame(payload.get("prediction_rows", []))
    if rows.empty:
        raise ValueError("No prediction rows found in model comparison payload.")
    if "model_name" not in rows.columns:
        raise ValueError("Prediction rows must include 'model_name'.")

    champion = rows[rows["model_name"] == champion_model].copy()
    if champion.empty:
        raise ValueError(f"No prediction rows found for champion model '{champion_model}'.")

    champion.loc[:, PROBABILITY_COLUMNS] = probabilities_to_unit_interval(champion.loc[:, PROBABILITY_COLUMNS])
    champion = champion.sort_values(_sort_columns(champion), kind="mergesort").reset_index(drop=True)
    return champion


def probabilities_to_unit_interval(probabilities: pd.DataFrame | list[list[float]] | np.ndarray) -> np.ndarray:
    """Convert percentage or unit-scale probabilities to validated 0-1 rows."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("Probabilities must be a 2D array with exactly three columns.")
    if np.nanmax(probs) > 1.0:
        probs = probs / 100.0
    return validate_probabilities(probs)


def temperature_scale_probabilities(
    probabilities: list[list[float]] | np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply multiclass temperature scaling to probability rows."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    probs = validate_probabilities(probabilities)
    logits = np.log(np.clip(probs, 1e-15, 1.0)) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def fit_temperature_shrinkage_calibrator(
    probabilities: list[list[float]] | np.ndarray,
    outcomes: list[str] | np.ndarray,
    *,
    temperature_grid: list[float] | np.ndarray | None = None,
    shrinkage_grid: list[float] | np.ndarray | None = None,
) -> TemperatureShrinkageCalibrator:
    """Fit simple calibration parameters by minimizing calibration-split log loss."""
    probs = validate_probabilities(probabilities)
    actual = outcome_indices(outcomes)
    if len(probs) != len(actual):
        raise ValueError("Probabilities and outcomes must have the same length.")

    temperatures = np.asarray(temperature_grid if temperature_grid is not None else np.linspace(0.7, 2.5, 73))
    shrinkages = np.asarray(shrinkage_grid if shrinkage_grid is not None else np.linspace(0.0, 0.35, 36))
    base_rates = one_hot_outcomes(actual).mean(axis=0)

    best_log_loss = float("inf")
    best_temperature = 1.0
    best_shrinkage = 0.0
    for temperature in temperatures:
        scaled = temperature_scale_probabilities(probs, float(temperature))
        for shrinkage in shrinkages:
            candidate = ((1.0 - float(shrinkage)) * scaled) + (float(shrinkage) * base_rates)
            log_loss = multiclass_log_loss(candidate, actual)
            if log_loss < best_log_loss:
                best_log_loss = log_loss
                best_temperature = float(temperature)
                best_shrinkage = float(shrinkage)

    return TemperatureShrinkageCalibrator(
        temperature=best_temperature,
        shrinkage=best_shrinkage,
        base_rates=tuple(float(value) for value in base_rates),
    )


def build_calibration_experiment(
    payload: dict[str, Any],
    *,
    champion_model: str = CHAMPION_MODEL,
    calibration_fraction: float = 0.6,
    n_bins: int = 5,
    top_n: int = 10,
) -> dict[str, Any]:
    """Fit on earlier champion folds and evaluate calibrated probabilities on later folds."""
    champion = extract_champion_predictions(payload, champion_model=champion_model)
    calibration_rows, trial_rows = split_by_time(champion, calibration_fraction=calibration_fraction)

    calibration_probabilities = probabilities_to_unit_interval(calibration_rows.loc[:, PROBABILITY_COLUMNS])
    calibration_outcomes = extract_outcomes(calibration_rows)
    calibrator = fit_temperature_shrinkage_calibrator(calibration_probabilities, calibration_outcomes)

    original_probabilities = probabilities_to_unit_interval(trial_rows.loc[:, PROBABILITY_COLUMNS])
    trial_outcomes = extract_outcomes(trial_rows)
    calibrated_probabilities = calibrator.transform(original_probabilities)

    original_metrics = metric_summary("original_champion", original_probabilities, trial_outcomes)
    calibrated_metrics = metric_summary("calibrated_champion", calibrated_probabilities, trial_outcomes)
    comparison = compare_metric_summaries(original_metrics, calibrated_metrics)

    calibrated_rows = trial_rows.copy()
    calibrated_rows.loc[:, PROBABILITY_COLUMNS] = calibrated_probabilities

    return {
        "champion_model": champion_model,
        "methodology": {
            "method": "Fit temperature scaling plus base-rate shrinkage on earlier folds; evaluate on later folds.",
            "calibration_fraction": calibration_fraction,
            "calibration_matches": int(len(calibration_rows)),
            "trial_matches": int(len(trial_rows)),
            "calibration_folds": _fold_labels(calibration_rows),
            "trial_folds": _fold_labels(trial_rows),
        },
        "parameters": calibrator.to_dict(),
        "overall_metrics": [original_metrics, calibrated_metrics],
        "metric_delta_calibrated_minus_original": comparison,
        "original_calibration": calibration_summary(original_probabilities, trial_outcomes, n_bins=n_bins, min_bin_size=2),
        "calibrated_calibration": calibration_summary(
            calibrated_probabilities,
            trial_outcomes,
            n_bins=n_bins,
            min_bin_size=2,
        ),
        "high_confidence_behaviour": {
            "original": high_confidence_summary(original_probabilities, trial_outcomes),
            "calibrated": high_confidence_summary(calibrated_probabilities, trial_outcomes),
        },
        "worst_misses_after_calibration": worst_misses(calibrated_rows, n=top_n),
        "interpretation": interpretation_lines(original_metrics, calibrated_metrics),
    }


def split_by_time(df: pd.DataFrame, *, calibration_fraction: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into earlier calibration and later trial folds."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")
    ordered = df.sort_values(_sort_columns(df), kind="mergesort").reset_index(drop=True)
    if "fold_id" in ordered.columns:
        fold_order = ordered["fold_id"].drop_duplicates().tolist()
        split_at = max(1, min(len(fold_order) - 1, int(round(len(fold_order) * calibration_fraction))))
        calibration_folds = set(fold_order[:split_at])
        calibration_rows = ordered[ordered["fold_id"].isin(calibration_folds)].copy()
        trial_rows = ordered[~ordered["fold_id"].isin(calibration_folds)].copy()
    else:
        split_at = max(1, min(len(ordered) - 1, int(round(len(ordered) * calibration_fraction))))
        calibration_rows = ordered.iloc[:split_at].copy()
        trial_rows = ordered.iloc[split_at:].copy()

    if calibration_rows.empty or trial_rows.empty:
        raise ValueError("Calibration experiment requires non-empty calibration and trial splits.")
    return calibration_rows.reset_index(drop=True), trial_rows.reset_index(drop=True)


def metric_summary(name: str, probabilities: list[list[float]] | np.ndarray, outcomes: list[str] | np.ndarray) -> dict[str, Any]:
    """Return core forecast metrics for one probability set."""
    return {
        "model_name": name,
        "n_matches": int(len(validate_probabilities(probabilities))),
        "brier_score": brier_score_3way(probabilities, outcomes),
        "log_loss": multiclass_log_loss(probabilities, outcomes),
        "accuracy": outcome_accuracy(probabilities, outcomes),
    }


def compare_metric_summaries(original: dict[str, Any], calibrated: dict[str, Any]) -> dict[str, Any]:
    """Return calibrated minus original metric deltas."""
    return {
        "brier_score": round(float(calibrated["brier_score"] - original["brier_score"]), 4),
        "log_loss": round(float(calibrated["log_loss"] - original["log_loss"]), 4),
        "accuracy": round(float(calibrated["accuracy"] - original["accuracy"]), 4),
    }


def high_confidence_summary(
    probabilities: list[list[float]] | np.ndarray,
    outcomes: list[str] | np.ndarray,
    *,
    threshold: float = 0.6,
) -> dict[str, Any]:
    """Summarise high-confidence top-class behaviour."""
    probs = validate_probabilities(probabilities)
    actual = outcome_indices(outcomes)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == actual
    mask = confidence >= threshold
    count = int(mask.sum())
    return {
        "threshold": threshold,
        "count": count,
        "share": round(float(count / len(probs)), 4),
        "accuracy": round(float(correct[mask].mean()), 4) if count else None,
        "mean_confidence": round(float(confidence[mask].mean()), 4) if count else None,
    }


def worst_misses(rows: pd.DataFrame, *, n: int = 10) -> list[dict[str, Any]]:
    """Return largest calibrated misses by row log loss."""
    scored = scored_prediction_rows(rows)
    columns = [
        "match_date",
        "round",
        "home_team",
        "away_team",
        "actual_outcome",
        "predicted_outcome",
        "predicted_confidence",
        "actual_probability",
        "row_log_loss",
    ]
    available = [column for column in columns if column in scored.columns]
    rows_out = scored.sort_values(["row_log_loss", "actual_probability"], ascending=[False, True], kind="mergesort").head(n)
    return _rounded_records(rows_out.loc[:, available])


def render_markdown_report(summary: dict[str, Any]) -> str:
    """Render the calibration experiment summary as Markdown."""
    delta = summary["metric_delta_calibrated_minus_original"]
    recommendation = _recommendation(summary)
    lines = [
        "# Champion Calibration Experiment",
        "",
        "## Executive Summary",
        "",
        (
            f"This offline experiment fits a calibration layer for `{summary['champion_model']}` on "
            f"{summary['methodology']['calibration_matches']} earlier matches and evaluates it on "
            f"{summary['methodology']['trial_matches']} later matches."
        ),
        (
            f"Calibrated minus original deltas on the trial split: Brier {delta['brier_score']:+.4f}, "
            f"log loss {delta['log_loss']:+.4f}, accuracy {delta['accuracy']:+.4f}."
        ),
        recommendation,
        "",
        "## Methodology",
        "",
        f"- {summary['methodology']['method']}",
        f"- Calibration folds: {', '.join(summary['methodology']['calibration_folds'])}",
        f"- Trial folds: {', '.join(summary['methodology']['trial_folds'])}",
        "- The underlying champion model, API, dashboard, and production outputs are unchanged.",
        "",
        "## Original Champion Vs Calibrated Champion",
        "",
        _markdown_table(
            summary["overall_metrics"],
            ["model_name", "n_matches", "brier_score", "log_loss", "accuracy"],
        ),
        "",
        "## Calibration Parameters",
        "",
        _markdown_table([_flatten_parameters(summary["parameters"])], ["method", "temperature", "shrinkage", "base_home", "base_draw", "base_away"]),
        "",
        "## Overall Metrics",
        "",
        _markdown_table(
            [summary["metric_delta_calibrated_minus_original"]],
            ["brier_score", "log_loss", "accuracy"],
        ),
        "",
        "## Calibration Bands",
        "",
        "### Original Champion",
        "",
        _markdown_table(
            summary["original_calibration"]["calibration_bins"],
            ["bin", "count", "mean_confidence", "observed_accuracy", "calibration_gap", "is_sparse"],
        ),
        "",
        "### Calibrated Champion",
        "",
        _markdown_table(
            summary["calibrated_calibration"]["calibration_bins"],
            ["bin", "count", "mean_confidence", "observed_accuracy", "calibration_gap", "is_sparse"],
        ),
        "",
        "## High-Confidence Behaviour",
        "",
        _markdown_table(
            [
                {"model_name": "original_champion", **summary["high_confidence_behaviour"]["original"]},
                {"model_name": "calibrated_champion", **summary["high_confidence_behaviour"]["calibrated"]},
            ],
            ["model_name", "threshold", "count", "share", "accuracy", "mean_confidence"],
        ),
        "",
        "## Worst Misses After Calibration",
        "",
        _markdown_table(
            summary["worst_misses_after_calibration"],
            [
                "match_date",
                "round",
                "home_team",
                "away_team",
                "actual_outcome",
                "predicted_outcome",
                "predicted_confidence",
                "actual_probability",
                "row_log_loss",
            ],
        ),
        "",
        "## Limitations",
        "",
        "- This is one offline experiment over the first comparison artefact.",
        "- Calibration parameters are fit on a small WSL sample, so they should be treated as hypotheses.",
        "- The trial split is later folds only, which is safer than evaluating on the same rows used for calibration.",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        *[f"- {line}" for line in summary["interpretation"]],
        "",
    ]
    return "\n".join(lines)


def interpretation_lines(original: dict[str, Any], calibrated: dict[str, Any]) -> list[str]:
    """Return cautious interpretation bullets for the trial result."""
    lines = []
    probability_quality_improved = calibrated["log_loss"] < original["log_loss"] or calibrated["brier_score"] < original[
        "brier_score"
    ]
    if probability_quality_improved:
        lines.append("Probability quality improved on at least one scoring metric, so this calibration is worth tracking.")
    else:
        lines.append("Calibration did not improve Brier score or log loss on this trial split; keep the original champion as reference.")
    if calibrated["accuracy"] < original["accuracy"] and probability_quality_improved:
        lines.append(
            "Accuracy fell, but calibration can still be useful when log loss or Brier improves because probability quality matters."
        )
    elif calibrated["accuracy"] < original["accuracy"]:
        lines.append("Accuracy also fell on this trial split, so this calibrated variant is not a production-testing candidate yet.")
    elif calibrated["accuracy"] == original["accuracy"]:
        lines.append("Top-class accuracy was unchanged; the experiment mainly changes probability quality.")
    else:
        lines.append("Top-class accuracy also improved on this trial split.")
    return lines


def _recommendation(summary: dict[str, Any]) -> str:
    delta = summary["metric_delta_calibrated_minus_original"]
    if delta["log_loss"] < 0 or delta["brier_score"] < 0:
        return (
            "Recommendation: consider the calibrated champion for future production testing, but keep the "
            "uncalibrated champion as the reference until it wins repeated backtests."
        )
    return "Recommendation: do not replace the champion reference from this experiment; keep testing calibration variants offline."


def _sort_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in ("fold_id", "match_date", "home_team", "away_team") if column in df.columns]


def _fold_labels(df: pd.DataFrame) -> list[str]:
    if "fold_id" not in df.columns:
        return []
    return df["fold_id"].drop_duplicates().astype(str).tolist()


def _flatten_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    base_rates = parameters["base_rates"]
    return {
        "method": parameters["method"],
        "temperature": parameters["temperature"],
        "shrinkage": parameters["shrinkage"],
        "base_home": base_rates["home_win"],
        "base_draw": base_rates["draw"],
        "base_away": base_rates["away_win"],
    }


def _rounded_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, float):
                record[key] = round(value, 4)
    return records


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    if not rows:
        return "\n".join([header, divider])
    body = ["| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
