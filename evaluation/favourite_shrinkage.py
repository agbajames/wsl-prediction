"""
Offline favourite-shrinkage experiments for champion 1X2 probabilities.

These helpers transform existing prediction rows only. They do not change the
production champion model, API, dashboard, or prediction outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.compare import PROBABILITY_COLUMNS, extract_outcomes
from evaluation.failure_analysis import scored_prediction_rows
from evaluation.metrics import brier_score_3way, multiclass_log_loss, outcome_accuracy, outcome_indices, validate_probabilities

CHAMPION_MODEL = "champion_dc_xg"


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
    """Return champion prediction rows with H/D/A probabilities normalized to 0-1."""
    rows = pd.DataFrame(payload.get("prediction_rows", []))
    if rows.empty:
        raise ValueError("No prediction rows found in model comparison payload.")
    if "model_name" not in rows.columns:
        raise ValueError("Prediction rows must include 'model_name'.")

    champion = rows[rows["model_name"] == champion_model].copy()
    if champion.empty:
        raise ValueError(f"No prediction rows found for champion model '{champion_model}'.")

    champion.loc[:, PROBABILITY_COLUMNS] = probabilities_to_unit_interval(champion.loc[:, PROBABILITY_COLUMNS])
    return champion.sort_values(_sort_columns(champion), kind="mergesort").reset_index(drop=True)


def probabilities_to_unit_interval(probabilities: pd.DataFrame | list[list[float]] | np.ndarray) -> np.ndarray:
    """Convert percentage or unit-scale probabilities to validated 0-1 rows."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("Probabilities must be a 2D array with exactly three columns.")
    if np.nanmax(probs) > 1.0:
        probs = probs / 100.0
    return validate_probabilities(probs)


def threshold_favourite_shrinkage(
    probabilities: list[list[float]] | np.ndarray,
    *,
    threshold: float,
    strength: float,
) -> np.ndarray:
    """Shrink favourites above a threshold and redistribute removed mass."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")

    probs = validate_probabilities(probabilities)
    adjusted = probs.copy()
    favourite_idx = adjusted.argmax(axis=1)
    favourite_prob = adjusted[np.arange(len(adjusted)), favourite_idx]
    shrink_mask = favourite_prob > threshold
    removed = np.minimum(strength, favourite_prob - threshold)
    removed = np.where(shrink_mask, removed, 0.0)
    adjusted[np.arange(len(adjusted)), favourite_idx] -= removed
    _redistribute_removed_mass(adjusted, favourite_idx, removed)
    return validate_probabilities(adjusted)


def soft_cap_favourite_shrinkage(
    probabilities: list[list[float]] | np.ndarray,
    *,
    cap: float,
    strength: float,
) -> np.ndarray:
    """Partly shrink favourites above a cap toward that cap."""
    if not 0.0 <= cap <= 1.0:
        raise ValueError("cap must be between 0 and 1.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")

    probs = validate_probabilities(probabilities)
    adjusted = probs.copy()
    favourite_idx = adjusted.argmax(axis=1)
    favourite_prob = adjusted[np.arange(len(adjusted)), favourite_idx]
    removed = np.where(favourite_prob > cap, (favourite_prob - cap) * strength, 0.0)
    adjusted[np.arange(len(adjusted)), favourite_idx] -= removed
    _redistribute_removed_mass(adjusted, favourite_idx, removed)
    return validate_probabilities(adjusted)


def default_variants() -> list[dict[str, Any]]:
    """Return fixed favourite-shrinkage variants for the first experiment."""
    variants = [{"variant_name": "original_champion", "method": "original", "threshold": None, "cap": None, "strength": 0.0}]
    for threshold in (0.65, 0.70, 0.75, 0.80):
        for strength in (0.05, 0.10, 0.15):
            variants.append(
                {
                    "variant_name": f"threshold_{threshold:.2f}_shrink_{strength:.2f}",
                    "method": "threshold",
                    "threshold": threshold,
                    "cap": None,
                    "strength": strength,
                }
            )
    for cap in (0.70, 0.75, 0.80):
        for strength in (0.25, 0.50, 0.75):
            variants.append(
                {
                    "variant_name": f"soft_cap_{cap:.2f}_strength_{strength:.2f}",
                    "method": "soft_cap",
                    "threshold": None,
                    "cap": cap,
                    "strength": strength,
                }
            )
    return variants


def apply_favourite_variant(probabilities: list[list[float]] | np.ndarray, variant: dict[str, Any]) -> np.ndarray:
    """Apply one favourite-shrinkage variant."""
    if variant["method"] == "original":
        return validate_probabilities(probabilities)
    if variant["method"] == "threshold":
        return threshold_favourite_shrinkage(
            probabilities,
            threshold=float(variant["threshold"]),
            strength=float(variant["strength"]),
        )
    if variant["method"] == "soft_cap":
        return soft_cap_favourite_shrinkage(
            probabilities,
            cap=float(variant["cap"]),
            strength=float(variant["strength"]),
        )
    raise ValueError(f"Unknown favourite-shrinkage method: {variant['method']}")


def build_favourite_shrinkage_experiment(
    payload: dict[str, Any],
    *,
    champion_model: str = CHAMPION_MODEL,
    high_confidence_threshold: float = 0.65,
    top_n: int = 10,
) -> dict[str, Any]:
    """Compare original champion probabilities with favourite-shrinkage variants."""
    champion_rows = extract_champion_predictions(payload, champion_model=champion_model)
    original_probabilities = probabilities_to_unit_interval(champion_rows.loc[:, PROBABILITY_COLUMNS])
    outcomes = extract_outcomes(champion_rows)

    variant_summaries = []
    adjusted_rows_by_name: dict[str, pd.DataFrame] = {}
    for variant in default_variants():
        adjusted_probabilities = apply_favourite_variant(original_probabilities, variant)
        metrics = metric_summary(variant["variant_name"], adjusted_probabilities, outcomes)
        favourite_metrics = high_confidence_favourite_metrics(
            adjusted_probabilities,
            outcomes,
            threshold=high_confidence_threshold,
        )
        summary = {**variant, **metrics, **favourite_metrics}
        variant_summaries.append(summary)

        adjusted_rows = champion_rows.copy()
        adjusted_rows.loc[:, PROBABILITY_COLUMNS] = adjusted_probabilities
        adjusted_rows_by_name[variant["variant_name"]] = adjusted_rows

    original = _find_variant(variant_summaries, "original_champion")
    best_by_log_loss = min(variant_summaries, key=lambda row: (row["log_loss"], row["brier_score"], row["variant_name"]))
    best_by_brier = min(variant_summaries, key=lambda row: (row["brier_score"], row["log_loss"], row["variant_name"]))
    best_rows = adjusted_rows_by_name[best_by_log_loss["variant_name"]]

    return {
        "champion_model": champion_model,
        "input_summary": {
            "n_matches": int(len(champion_rows)),
            "high_confidence_threshold": high_confidence_threshold,
        },
        "methodology": {
            "method": "Apply fixed threshold and soft-cap shrinkage to high-confidence favourites.",
            "variants_tested": len(variant_summaries),
            "thresholds": [0.65, 0.70, 0.75, 0.80],
            "threshold_strengths": [0.05, 0.10, 0.15],
            "soft_caps": [0.70, 0.75, 0.80],
            "soft_cap_strengths": [0.25, 0.50, 0.75],
        },
        "original_favourite_behaviour": high_confidence_favourite_metrics(
            original_probabilities,
            outcomes,
            threshold=high_confidence_threshold,
        ),
        "home_away_favourite_behaviour": home_away_favourite_summary(original_probabilities, outcomes),
        "variants": [_rounded_record(row) for row in variant_summaries],
        "best_by_log_loss": _rounded_record(best_by_log_loss),
        "best_by_brier": _rounded_record(best_by_brier),
        "best_vs_original_delta": compare_variant_metrics(original, best_by_log_loss),
        "worst_misses_after_shrinkage": worst_misses(best_rows, n=top_n),
        "interpretation": interpretation_lines(original, best_by_log_loss),
    }


def metric_summary(name: str, probabilities: list[list[float]] | np.ndarray, outcomes: list[str] | np.ndarray) -> dict[str, Any]:
    """Return overall metrics for one variant."""
    return {
        "model_name": name,
        "n_matches": int(len(validate_probabilities(probabilities))),
        "brier_score": brier_score_3way(probabilities, outcomes),
        "log_loss": multiclass_log_loss(probabilities, outcomes),
        "accuracy": outcome_accuracy(probabilities, outcomes),
    }


def high_confidence_favourite_metrics(
    probabilities: list[list[float]] | np.ndarray,
    outcomes: list[str] | np.ndarray,
    *,
    threshold: float = 0.65,
) -> dict[str, Any]:
    """Return favourite-focused metrics above the confidence threshold."""
    probs = validate_probabilities(probabilities)
    actual = outcome_indices(outcomes)
    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predicted == actual
    high = confidence >= threshold
    high_miss = high & ~correct
    high_correct = high & correct
    return {
        "high_confidence_threshold": threshold,
        "high_confidence_favourites": int(high.sum()),
        "high_confidence_favourite_accuracy": round(float(correct[high].mean()), 4) if high.any() else None,
        "mean_favourite_confidence": round(float(confidence[high].mean()), 4) if high.any() else None,
        "high_confidence_miss_count": int(high_miss.sum()),
        "high_confidence_correct_count": int(high_correct.sum()),
        "high_confidence_miss_log_loss": _masked_log_loss(probs, actual, high_miss),
        "high_confidence_correct_log_loss": _masked_log_loss(probs, actual, high_correct),
    }


def home_away_favourite_summary(probabilities: list[list[float]] | np.ndarray, outcomes: list[str] | np.ndarray) -> list[dict[str, Any]]:
    """Summarise home favourite, away favourite, and draw-favourite behaviour."""
    probs = validate_probabilities(probabilities)
    actual = outcome_indices(outcomes)
    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    labels = {0: "home_favourite", 1: "draw_favourite", 2: "away_favourite"}
    rows = []
    for idx, label in labels.items():
        mask = predicted == idx
        rows.append(
            {
                "favourite_type": label,
                "n": int(mask.sum()),
                "accuracy": round(float((predicted[mask] == actual[mask]).mean()), 4) if mask.any() else None,
                "mean_confidence": round(float(confidence[mask].mean()), 4) if mask.any() else None,
                "mean_log_loss": _masked_log_loss(probs, actual, mask),
            }
        )
    return rows


def compare_variant_metrics(original: dict[str, Any], adjusted: dict[str, Any]) -> dict[str, Any]:
    """Return adjusted minus original metric deltas."""
    return {
        "variant_name": adjusted["variant_name"],
        "brier_score": round(float(adjusted["brier_score"] - original["brier_score"]), 4),
        "log_loss": round(float(adjusted["log_loss"] - original["log_loss"]), 4),
        "accuracy": round(float(adjusted["accuracy"] - original["accuracy"]), 4),
        "high_confidence_miss_count": int(
            adjusted["high_confidence_miss_count"] - original["high_confidence_miss_count"]
        ),
        "high_confidence_correct_count": int(
            adjusted["high_confidence_correct_count"] - original["high_confidence_correct_count"]
        ),
        "mean_favourite_confidence": round(
            float(adjusted["mean_favourite_confidence"] - original["mean_favourite_confidence"]),
            4,
        ),
    }


def worst_misses(rows: pd.DataFrame, *, n: int = 10) -> list[dict[str, Any]]:
    """Return the largest misses for adjusted rows by row-level log loss."""
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
    return [_rounded_record(row) for row in rows_out.loc[:, available].to_dict(orient="records")]


def render_markdown_report(summary: dict[str, Any]) -> str:
    """Render the favourite-shrinkage experiment summary as Markdown."""
    best = summary["best_by_log_loss"]
    delta = summary["best_vs_original_delta"]
    recommendation = _recommendation(summary)
    lines = [
        "# Champion Favourite-Shrinkage Experiment",
        "",
        "## Executive Summary",
        "",
        (
            f"This offline experiment tests favourite shrinkage for `{summary['champion_model']}` on "
            f"{summary['input_summary']['n_matches']} evaluated matches."
        ),
        (
            f"Best variant by log loss: `{best['variant_name']}`. Deltas versus original: "
            f"Brier {delta['brier_score']:+.4f}, log loss {delta['log_loss']:+.4f}, "
            f"accuracy {delta['accuracy']:+.4f}."
        ),
        recommendation,
        "",
        "## Methodology",
        "",
        f"- {summary['methodology']['method']}",
        f"- Thresholds tested: {', '.join(str(value) for value in summary['methodology']['thresholds'])}",
        f"- Threshold shrink strengths tested: {', '.join(str(value) for value in summary['methodology']['threshold_strengths'])}",
        f"- Soft caps tested: {', '.join(str(value) for value in summary['methodology']['soft_caps'])}",
        f"- Soft-cap strengths tested: {', '.join(str(value) for value in summary['methodology']['soft_cap_strengths'])}",
        "- The underlying champion model, API, dashboard, and production outputs are unchanged.",
        "",
        "## Original Champion Favourite Behaviour",
        "",
        _markdown_table([summary["original_favourite_behaviour"]], _favourite_metric_columns()),
        "",
        "## Favourite-Shrinkage Variants Tested",
        "",
        _markdown_table(
            summary["variants"],
            [
                "variant_name",
                "method",
                "threshold",
                "cap",
                "strength",
                "brier_score",
                "log_loss",
                "accuracy",
                "high_confidence_favourites",
                "high_confidence_miss_count",
                "high_confidence_correct_count",
            ],
        ),
        "",
        "## Original Champion Vs Best Shrinkage Variant",
        "",
        _markdown_table(
            [_find_variant(summary["variants"], "original_champion"), best],
            ["variant_name", "brier_score", "log_loss", "accuracy", *_favourite_metric_columns()[1:]],
        ),
        "",
        "## Overall Metrics",
        "",
        _markdown_table(
            [summary["best_vs_original_delta"]],
            [
                "variant_name",
                "brier_score",
                "log_loss",
                "accuracy",
                "high_confidence_miss_count",
                "high_confidence_correct_count",
                "mean_favourite_confidence",
            ],
        ),
        "",
        "## High-Confidence Favourite Metrics",
        "",
        _markdown_table(summary["variants"], ["variant_name", *_favourite_metric_columns()]),
        "",
        "## Home Favourite Vs Away Favourite Behaviour",
        "",
        _markdown_table(
            summary["home_away_favourite_behaviour"],
            ["favourite_type", "n", "accuracy", "mean_confidence", "mean_log_loss"],
        ),
        "",
        "## Worst Misses After Shrinkage",
        "",
        _markdown_table(
            summary["worst_misses_after_shrinkage"],
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
        "- This is one offline experiment over one generated comparison artefact.",
        "- The sample contains 109 champion-evaluated matches, so differences should be treated as hypotheses.",
        "- Fixed shrinkage variants are intentionally simple and do not learn a new production model.",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        *[f"- {line}" for line in summary["interpretation"]],
        "",
    ]
    return "\n".join(lines)


def interpretation_lines(original: dict[str, Any], best: dict[str, Any]) -> list[str]:
    """Return cautious interpretation bullets for the best shrinkage variant."""
    improved_quality = best["log_loss"] < original["log_loss"] or best["brier_score"] < original["brier_score"]
    reduced_misses = best["high_confidence_miss_count"] < original["high_confidence_miss_count"]
    damaged_correct = best["high_confidence_correct_count"] < original["high_confidence_correct_count"]
    lines = []
    if improved_quality:
        lines.append("The best favourite-shrinkage variant improved at least one probability-quality metric.")
    else:
        lines.append("No favourite-shrinkage variant improved log loss or Brier score over the original champion.")
    if reduced_misses:
        lines.append("High-confidence miss count fell for the best variant.")
    else:
        lines.append("High-confidence miss count did not fall for the best variant.")
    if damaged_correct:
        lines.append("The variant also reduced high-confidence correct calls, so the tradeoff needs shadow testing before promotion.")
    if best["accuracy"] < original["accuracy"] and improved_quality:
        lines.append(
            "Accuracy fell, but this can still be useful when Brier or log loss improves because probability quality matters."
        )
    elif best["accuracy"] < original["accuracy"]:
        lines.append("Accuracy fell alongside probability quality, so the original champion should remain unchanged.")
    elif best["variant_name"] == original["variant_name"]:
        lines.append("The original champion remains the best tested option, so no shrinkage variant should be promoted.")
    elif best["accuracy"] == original["accuracy"]:
        lines.append("Top-class accuracy was unchanged; the experiment mainly changed probability sharpness.")
    else:
        lines.append("Top-class accuracy improved for the best variant.")
    return lines


def _redistribute_removed_mass(adjusted: np.ndarray, favourite_idx: np.ndarray, removed: np.ndarray) -> None:
    for row_idx, amount in enumerate(removed):
        if amount <= 0:
            continue
        other_indices = [idx for idx in range(3) if idx != favourite_idx[row_idx]]
        other_total = adjusted[row_idx, other_indices].sum()
        if other_total <= 0:
            adjusted[row_idx, other_indices] += amount / 2.0
        else:
            adjusted[row_idx, other_indices] += amount * (adjusted[row_idx, other_indices] / other_total)


def _recommendation(summary: dict[str, Any]) -> str:
    original = _find_variant(summary["variants"], "original_champion")
    best = summary["best_by_log_loss"]
    if best["variant_name"] == "original_champion":
        return "Recommendation: keep the original champion unchanged; no tested favourite shrinkage improved log loss."
    if best["log_loss"] < original["log_loss"] or best["brier_score"] < original["brier_score"]:
        return (
            "Recommendation: consider this favourite-shrinkage variant for future shadow testing, "
            "while keeping the original champion as the reference."
        )
    return "Recommendation: keep the original champion unchanged; tested shrinkage variants did not improve quality."


def _masked_log_loss(probs: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    one_hot = np.zeros((int(mask.sum()), 3), dtype=float)
    one_hot[np.arange(int(mask.sum())), actual[mask]] = 1.0
    clipped = np.clip(probs[mask], 1e-15, 1.0)
    return round(float(-np.mean(np.sum(one_hot * np.log(clipped), axis=1))), 4)


def _find_variant(variants: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    for variant in variants:
        if variant["variant_name"] == variant_name:
            return variant
    raise ValueError(f"Variant not found: {variant_name}")


def _sort_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in ("fold_id", "match_date", "home_team", "away_team") if column in df.columns]


def _favourite_metric_columns() -> list[str]:
    return [
        "high_confidence_threshold",
        "high_confidence_favourites",
        "high_confidence_favourite_accuracy",
        "mean_favourite_confidence",
        "high_confidence_miss_count",
        "high_confidence_correct_count",
        "high_confidence_miss_log_loss",
        "high_confidence_correct_log_loss",
    ]


def _rounded_record(row: dict[str, Any]) -> dict[str, Any]:
    rounded = dict(row)
    for key, value in list(rounded.items()):
        if isinstance(value, float):
            rounded[key] = round(value, 4)
    return rounded


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
