"""
Offline draw-adjustment experiments for champion 1X2 probabilities.

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
from evaluation.metrics import (
    OUTCOME_LABELS,
    brier_score_3way,
    multiclass_log_loss,
    outcome_accuracy,
    outcome_indices,
    validate_probabilities,
)

CHAMPION_MODEL = "champion_dc_xg"
DRAW_COLUMN_INDEX = 1


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


def additive_draw_adjustment(probabilities: list[list[float]] | np.ndarray, delta: float) -> np.ndarray:
    """Add draw mass and remove it proportionally from home/away probabilities."""
    probs = validate_probabilities(probabilities)
    adjusted = probs.copy()
    new_draw = np.clip(adjusted[:, DRAW_COLUMN_INDEX] + delta, 0.0, 1.0)
    home_away_total = adjusted[:, 0] + adjusted[:, 2]
    remaining_mass = 1.0 - new_draw

    home_share = np.divide(
        adjusted[:, 0],
        home_away_total,
        out=np.full(len(adjusted), 0.5),
        where=home_away_total > 0,
    )
    adjusted[:, 0] = remaining_mass * home_share
    adjusted[:, DRAW_COLUMN_INDEX] = new_draw
    adjusted[:, 2] = remaining_mass * (1.0 - home_share)
    return validate_probabilities(adjusted)


def multiplicative_draw_adjustment(probabilities: list[list[float]] | np.ndarray, factor: float) -> np.ndarray:
    """Multiply draw probability by factor and renormalize each H/D/A row."""
    if factor < 0:
        raise ValueError("factor must be non-negative.")
    adjusted = validate_probabilities(probabilities).copy()
    adjusted[:, DRAW_COLUMN_INDEX] *= factor
    return validate_probabilities(adjusted)


def default_variants() -> list[dict[str, Any]]:
    """Return the simple draw-adjustment variants for the first experiment."""
    variants = [{"variant_name": "original_champion", "method": "original", "value": 0.0}]
    variants.extend(
        {"variant_name": f"additive_draw_{delta:+.3f}", "method": "additive", "value": delta}
        for delta in (-0.05, -0.025, 0.025, 0.05)
    )
    variants.extend(
        {"variant_name": f"multiplicative_draw_{factor:.2f}", "method": "multiplicative", "value": factor}
        for factor in (0.85, 0.95, 1.05, 1.15)
    )
    return variants


def apply_draw_variant(probabilities: list[list[float]] | np.ndarray, variant: dict[str, Any]) -> np.ndarray:
    """Apply one draw-adjustment variant to probability rows."""
    method = variant["method"]
    if method == "original":
        return validate_probabilities(probabilities)
    if method == "additive":
        return additive_draw_adjustment(probabilities, float(variant["value"]))
    if method == "multiplicative":
        return multiplicative_draw_adjustment(probabilities, float(variant["value"]))
    raise ValueError(f"Unknown draw-adjustment method: {method}")


def build_draw_adjustment_experiment(
    payload: dict[str, Any],
    *,
    champion_model: str = CHAMPION_MODEL,
    top_n: int = 10,
) -> dict[str, Any]:
    """Compare original champion probabilities with simple draw-adjusted variants."""
    champion_rows = extract_champion_predictions(payload, champion_model=champion_model)
    original_probabilities = probabilities_to_unit_interval(champion_rows.loc[:, PROBABILITY_COLUMNS])
    outcomes = extract_outcomes(champion_rows)

    variant_summaries = []
    adjusted_rows_by_name: dict[str, pd.DataFrame] = {}
    for variant in default_variants():
        adjusted_probabilities = apply_draw_variant(original_probabilities, variant)
        metrics = metric_summary(variant["variant_name"], adjusted_probabilities, outcomes)
        draw_metrics = draw_specific_metrics(adjusted_probabilities, outcomes)
        summary = {**variant, **metrics, **draw_metrics}
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
            "n_actual_draws": int(sum(outcome == "D" for outcome in outcomes)),
        },
        "methodology": {
            "method": "Apply fixed additive and multiplicative draw adjustments to champion probabilities.",
            "variants_tested": len(variant_summaries),
            "additive_deltas": [-0.05, -0.025, 0.025, 0.05],
            "multiplicative_factors": [0.85, 0.95, 1.05, 1.15],
        },
        "original_draw_behaviour": draw_specific_metrics(original_probabilities, outcomes),
        "variants": [_rounded_record(row) for row in variant_summaries],
        "best_by_log_loss": _rounded_record(best_by_log_loss),
        "best_by_brier": _rounded_record(best_by_brier),
        "best_vs_original_delta": compare_variant_metrics(original, best_by_log_loss),
        "worst_misses_after_draw_adjustment": worst_misses(best_rows, n=top_n),
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


def draw_specific_metrics(probabilities: list[list[float]] | np.ndarray, outcomes: list[str] | np.ndarray) -> dict[str, Any]:
    """Return draw-focused and non-draw performance metrics."""
    probs = validate_probabilities(probabilities)
    actual = outcome_indices(outcomes)
    predicted = probs.argmax(axis=1)
    actual_draw = actual == DRAW_COLUMN_INDEX
    predicted_draw = predicted == DRAW_COLUMN_INDEX
    non_draw = ~actual_draw

    return {
        "actual_draws": int(actual_draw.sum()),
        "draw_prediction_rate": round(float(predicted_draw.mean()), 4),
        "avg_predicted_draw_probability": round(float(probs[:, DRAW_COLUMN_INDEX].mean()), 4),
        "draw_recall": round(float(predicted_draw[actual_draw].mean()), 4) if actual_draw.any() else None,
        "draw_log_loss": _masked_log_loss(probs, actual, actual_draw),
        "non_draw_log_loss": _masked_log_loss(probs, actual, non_draw),
        "non_draw_accuracy": round(float((predicted[non_draw] == actual[non_draw]).mean()), 4) if non_draw.any() else None,
    }


def compare_variant_metrics(original: dict[str, Any], adjusted: dict[str, Any]) -> dict[str, Any]:
    """Return adjusted minus original metric deltas."""
    return {
        "variant_name": adjusted["variant_name"],
        "brier_score": round(float(adjusted["brier_score"] - original["brier_score"]), 4),
        "log_loss": round(float(adjusted["log_loss"] - original["log_loss"]), 4),
        "accuracy": round(float(adjusted["accuracy"] - original["accuracy"]), 4),
        "draw_recall": round(float(adjusted["draw_recall"] - original["draw_recall"]), 4),
        "draw_log_loss": round(float(adjusted["draw_log_loss"] - original["draw_log_loss"]), 4),
        "non_draw_log_loss": round(float(adjusted["non_draw_log_loss"] - original["non_draw_log_loss"]), 4),
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
    """Render the draw-adjustment experiment summary as Markdown."""
    best = summary["best_by_log_loss"]
    delta = summary["best_vs_original_delta"]
    recommendation = _recommendation(summary)
    lines = [
        "# Champion Draw-Adjustment Experiment",
        "",
        "## Executive Summary",
        "",
        (
            f"This offline experiment tests fixed draw adjustments for `{summary['champion_model']}` on "
            f"{summary['input_summary']['n_matches']} evaluated matches with "
            f"{summary['input_summary']['n_actual_draws']} actual draws."
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
        f"- Additive deltas tested: {', '.join(str(value) for value in summary['methodology']['additive_deltas'])}",
        (
            "- Multiplicative factors tested: "
            + ", ".join(str(value) for value in summary["methodology"]["multiplicative_factors"])
        ),
        "- The underlying champion model, API, dashboard, and production outputs are unchanged.",
        "",
        "## Original Champion Draw Behaviour",
        "",
        _markdown_table(
            [summary["original_draw_behaviour"]],
            [
                "actual_draws",
                "draw_prediction_rate",
                "avg_predicted_draw_probability",
                "draw_recall",
                "draw_log_loss",
                "non_draw_log_loss",
                "non_draw_accuracy",
            ],
        ),
        "",
        "## Draw-Adjustment Variants Tested",
        "",
        _markdown_table(
            summary["variants"],
            [
                "variant_name",
                "method",
                "value",
                "brier_score",
                "log_loss",
                "accuracy",
                "draw_prediction_rate",
                "avg_predicted_draw_probability",
                "draw_recall",
                "draw_log_loss",
                "non_draw_log_loss",
            ],
        ),
        "",
        "## Original Champion Vs Best Draw-Adjusted Variant",
        "",
        _markdown_table(
            [_find_variant(summary["variants"], "original_champion"), best],
            ["variant_name", "brier_score", "log_loss", "accuracy", "draw_recall", "draw_log_loss"],
        ),
        "",
        "## Overall Metrics",
        "",
        _markdown_table(
            [summary["best_vs_original_delta"]],
            ["variant_name", "brier_score", "log_loss", "accuracy", "draw_recall", "draw_log_loss"],
        ),
        "",
        "## Draw-Specific Metrics",
        "",
        _markdown_table(
            summary["variants"],
            [
                "variant_name",
                "actual_draws",
                "draw_prediction_rate",
                "avg_predicted_draw_probability",
                "draw_recall",
                "draw_log_loss",
                "non_draw_log_loss",
                "non_draw_accuracy",
            ],
        ),
        "",
        "## Worst Misses After Draw Adjustment",
        "",
        _markdown_table(
            summary["worst_misses_after_draw_adjustment"],
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
        "- Fixed adjustments are intentionally simple and do not learn a new production model.",
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
    """Return cautious interpretation bullets for the best draw variant."""
    improved_quality = best["log_loss"] < original["log_loss"] or best["brier_score"] < original["brier_score"]
    improved_draw_recall = best["draw_recall"] > original["draw_recall"]
    worsened_draw_log_loss = best["draw_log_loss"] > original["draw_log_loss"]
    lowered_draw_probability = best["avg_predicted_draw_probability"] < original["avg_predicted_draw_probability"]
    lines = []
    if improved_quality:
        lines.append("The best draw-adjusted variant improved at least one probability-quality metric.")
        if lowered_draw_probability:
            lines.append(
                "The winning variant shrinks draw probability, suggesting the champion may over-price draws on non-draws in this artefact."
            )
    else:
        lines.append("No draw-adjusted variant improved log loss or Brier score over the original champion.")
    if improved_draw_recall and not improved_quality:
        lines.append("Draw recall improved, but the overall probability metrics were worse, so this is not enough on its own.")
    elif improved_draw_recall:
        lines.append("Draw recall also improved, making this variant worth tracking in future offline tests.")
    elif worsened_draw_log_loss:
        lines.append("Actual-draw handling got worse, so the overall gain comes from non-draw fixtures rather than better draw recognition.")
    if best["accuracy"] < original["accuracy"] and improved_quality:
        lines.append(
            "Accuracy fell, but this can still be useful when Brier or log loss improves because probability quality matters."
        )
    elif best["accuracy"] < original["accuracy"]:
        lines.append("Accuracy fell alongside probability quality, so the original champion should remain unchanged.")
    elif best["accuracy"] == original["accuracy"]:
        lines.append("Top-class accuracy was unchanged; the experiment mainly changed probability quality.")
    else:
        lines.append("Top-class accuracy improved for the best variant.")
    return lines


def _recommendation(summary: dict[str, Any]) -> str:
    original = _find_variant(summary["variants"], "original_champion")
    best = summary["best_by_log_loss"]
    if best["variant_name"] == "original_champion":
        return "Recommendation: keep the original champion unchanged; no tested draw adjustment improved log loss."
    if best["log_loss"] < original["log_loss"] or best["brier_score"] < original["brier_score"]:
        return (
            "Recommendation: consider this draw-adjusted variant for future offline production-style testing, "
            "while keeping the original champion as the reference."
        )
    return "Recommendation: keep the original champion unchanged; tested draw adjustments did not improve quality."


def _masked_log_loss(probs: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    labels = [OUTCOME_LABELS[idx] for idx in actual[mask]]
    return round(float(multiclass_log_loss(probs[mask], labels)), 4)


def _find_variant(variants: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    for variant in variants:
        if variant["variant_name"] == variant_name:
            return variant
    raise ValueError(f"Variant not found: {variant_name}")


def _sort_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in ("fold_id", "match_date", "home_team", "away_team") if column in df.columns]


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
