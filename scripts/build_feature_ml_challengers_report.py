#!/usr/bin/env python3
"""
Build the consolidated Phase 8B feature-based ML challengers report.

The script reads committed first-run JSON evaluation artifacts and writes a
compact Markdown report plus a small JSON summary. It does not run models and
does not require live Supabase access.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PHASE_8A_SUMMARY = Path("reports/statistical_challengers_phase_8a_summary.json")
TREE_REPORT = Path("reports/tree_based_challenger_first_run.json")
IMPROVED_LOGISTIC_ABLATION = Path("reports/improved_logistic_regression_ablation_first_run.json")
TREE_IMPORTANCE_MD = Path("reports/tree_based_challenger_feature_importance_first_run.md")

OUTPUT_MD = Path("reports/feature_ml_challengers_phase_8b_summary.md")
OUTPUT_JSON = Path("reports/feature_ml_challengers_phase_8b_summary.json")

PRIMARY_MODELS = (
    "dc_fit_rho_each_fold",
    "txg_xg_pseudocount_010",
    "champion_dc_xg",
    "regularised_team_strength",
    "improved_logistic_regression",
    "random_forest",
    "logistic_regression",
    "poisson_regression",
    "elo_baseline",
    "naive_outcome_rate",
)

FEATURE_ML_MODELS = {
    "improved_logistic_regression",
    "random_forest",
    "logistic_regression",
}

MODEL_NOTES = {
    "dc_fit_rho_each_fold": "Phase 8A best probability-quality candidate; not promoted.",
    "txg_xg_pseudocount_010": "Phase 8A most balanced champion-family candidate; not promoted.",
    "champion_dc_xg": "Unchanged operational/reference model.",
    "regularised_team_strength": "Strongest standalone statistical challenger.",
    "improved_logistic_regression": "Best feature-based ML challenger on Brier score and log loss.",
    "random_forest": "Conservative tree-based challenger; best feature-ML accuracy in Phase 8B.",
    "logistic_regression": "Original feature-based ML baseline.",
    "poisson_regression": "Interpretable statistical challenger.",
    "elo_baseline": "Simple rating baseline.",
    "naive_outcome_rate": "Sanity-check baseline.",
}


def main() -> None:
    records = load_records()
    primary = [records[name] for name in PRIMARY_MODELS if name in records]
    ablations = load_ablation_rows()
    payload = build_payload(primary, ablations)
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_records() -> dict[str, dict[str, Any]]:
    """Load Phase 8A carried-forward rows plus Phase 8B comparison rows."""
    records: dict[str, dict[str, Any]] = {}

    phase_8a = json.loads(PHASE_8A_SUMMARY.read_text(encoding="utf-8"))
    for row in phase_8a["primary_models"]:
        records[row["model_name"]] = _record(row, PHASE_8A_SUMMARY)

    tree_report = json.loads(TREE_REPORT.read_text(encoding="utf-8"))
    for row in tree_report["report_summary"]["comparison"]:
        if row["model_name"] not in records or row["model_name"] in FEATURE_ML_MODELS:
            records[row["model_name"]] = _record(row, TREE_REPORT)

    return records


def load_ablation_rows() -> list[dict[str, Any]]:
    """Load improved-logistic feature-group ablations."""
    data = json.loads(IMPROVED_LOGISTIC_ABLATION.read_text(encoding="utf-8"))
    return [_record(row, IMPROVED_LOGISTIC_ABLATION) for row in data["report_summary"]["comparison"]]


def build_payload(primary: list[dict[str, Any]], ablations: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a serializable Phase 8B summary payload."""
    brier_ranking = sorted(primary, key=lambda row: row["brier_score"])
    log_loss_ranking = sorted(primary, key=lambda row: row["log_loss"])
    accuracy_ranking = sorted(primary, key=lambda row: row["accuracy"], reverse=True)
    feature_ml = [row for row in primary if row["model_name"] in FEATURE_ML_MODELS]
    feature_ml_brier = sorted(feature_ml, key=lambda row: row["brier_score"])
    feature_ml_accuracy = sorted(feature_ml, key=lambda row: row["accuracy"], reverse=True)
    return {
        "title": "Phase 8B Consolidated Feature-Based ML Challengers Report",
        "source_reports": {
            "phase_8a_summary": PHASE_8A_SUMMARY.as_posix(),
            "tree_based_comparison": TREE_REPORT.as_posix(),
            "improved_logistic_ablation": IMPROVED_LOGISTIC_ABLATION.as_posix(),
            "tree_feature_importance": TREE_IMPORTANCE_MD.as_posix(),
        },
        "primary_models": primary,
        "rankings": {
            "brier_score": brier_ranking,
            "log_loss": log_loss_ranking,
            "accuracy": accuracy_ranking,
        },
        "feature_ml_rankings": {
            "brier_score": feature_ml_brier,
            "accuracy": feature_ml_accuracy,
        },
        "improved_logistic_ablation": ablations,
        "decision": {
            "operational_reference": _find(primary, "champion_dc_xg"),
            "best_overall_probability_candidate": _find(primary, "dc_fit_rho_each_fold"),
            "most_balanced_champion_family_candidate": _find(primary, "txg_xg_pseudocount_010"),
            "strongest_standalone_statistical_challenger": _find(primary, "regularised_team_strength"),
            "best_feature_ml_probability_candidate": feature_ml_brier[0],
            "best_feature_ml_accuracy_candidate": feature_ml_accuracy[0],
            "production_promotion": "none",
            "recommendation": "Prioritise Phase 8D ensemble/blending and Phase 8E shadow testing; keep Phase 8C neural network work research-only or parked.",
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the consolidated Phase 8B report."""
    decision = payload["decision"]
    lines = [
        "# Phase 8B Consolidated Feature-Based ML Challengers Report",
        "",
        "## Scope",
        "",
        "This report consolidates Phase 8B feature-based ML challenger results and compares them",
        "with the Phase 8A statistical/champion-family leaders. It is documentation/reporting only",
        "and does not change production prediction behaviour.",
        "",
        "All headline rows use the same local 2025-10-01 to 2026-05-16 comparison window where available,",
        "with 109 evaluated matches. Lower Brier score and log loss are better; higher accuracy is better.",
        "",
        "## Primary Model Summary",
        "",
        _markdown_table(
            payload["primary_models"],
            ["model_name", "n_matches", "brier_score", "log_loss", "accuracy", "note"],
        ),
        "",
        "## Ranking By Brier Score",
        "",
        _markdown_table(
            _ranked(payload["rankings"]["brier_score"]),
            ["rank", "model_name", "brier_score", "log_loss", "accuracy"],
        ),
        "",
        "## Ranking By Log Loss",
        "",
        _markdown_table(
            _ranked(payload["rankings"]["log_loss"]),
            ["rank", "model_name", "brier_score", "log_loss", "accuracy"],
        ),
        "",
        "## Ranking By Accuracy",
        "",
        _markdown_table(
            _ranked(payload["rankings"]["accuracy"]),
            ["rank", "model_name", "brier_score", "log_loss", "accuracy"],
        ),
        "",
        "## Phase 8B Interpretation",
        "",
        f"- Best feature-based ML model by probability quality: `{decision['best_feature_ml_probability_candidate']['model_name']}` "
        f"with Brier {decision['best_feature_ml_probability_candidate']['brier_score']:.4f} and "
        f"log loss {decision['best_feature_ml_probability_candidate']['log_loss']:.4f}.",
        f"- Best feature-based ML model by accuracy: `{decision['best_feature_ml_accuracy_candidate']['model_name']}` "
        f"with accuracy {decision['best_feature_ml_accuracy_candidate']['accuracy']:.4f}.",
        "- Tree-based modelling added value over the original logistic baseline on log loss and accuracy,",
        "  but it did not beat improved logistic regression on Brier score or log loss.",
        "- Feature-based ML improved the ML baselines, but still trails the statistical/champion-family leaders.",
        "",
        "## Feature And Ablation Notes",
        "",
        "- The improved logistic `xg` feature group was the best ablation and became the registered default.",
        "- The full improved-logistic feature set underperformed, which is a useful warning about feature richness",
        "  on a one-season WSL sample.",
        "- The random-forest feature-importance artifact is available at",
        f"  `{payload['source_reports']['tree_feature_importance']}`.",
        "",
        "### Improved Logistic Ablation",
        "",
        _markdown_table(
            payload["improved_logistic_ablation"],
            ["model_name", "n_matches", "brier_score", "log_loss", "accuracy"],
        ),
        "",
        "## Model Decision",
        "",
        "- No production promotion yet.",
        "- Keep `champion_dc_xg` as the operational/reference model.",
        "- Carry `dc_fit_rho_each_fold` forward as the best overall probability-quality candidate.",
        "- Carry `txg_xg_pseudocount_010` forward as the most balanced champion-family candidate.",
        "- Carry `regularised_team_strength` forward as the strongest standalone statistical challenger.",
        "- Carry `improved_logistic_regression` and `random_forest` forward as Phase 8B ML benchmarks.",
        "",
        "## Next-Step Recommendation",
        "",
        "Recommendation: prioritise Phase 8D ensemble/blending and Phase 8E shadow testing before any",
        "production model decision. Phase 8C neural-network work should be parked or kept explicitly",
        "research-only because one WSL season is too small for a high-capacity model to earn promotion.",
        "",
        "## Limitations",
        "",
        "- The main comparison covers one WSL season and 109 evaluated matches.",
        "- The WSL dataset is small, so small metric differences can be unstable.",
        "- Feature-rich models remain vulnerable to overfitting even with leakage-safe feature generation.",
        "- Candidate models need shadow/live-style validation before any production decision.",
        "",
        "## Source Artifacts",
        "",
    ]
    for source_name, source_path in payload["source_reports"].items():
        lines.append(f"- `{source_name}`: `{source_path}`")
    lines.append("")
    return "\n".join(lines)


def _record(row: dict[str, Any], source_path: Path) -> dict[str, Any]:
    model_name = row["model_name"]
    return {
        "model_name": model_name,
        "n_matches": int(row["n_matches"]),
        "brier_score": float(row["brier_score"]),
        "log_loss": float(row["log_loss"]),
        "accuracy": float(row["accuracy"]),
        "source_report": source_path.as_posix(),
        "note": MODEL_NOTES.get(model_name, ""),
    }


def _find(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    for row in rows:
        if row["model_name"] == model_name:
            return row
    raise ValueError(f"Missing expected model row: {model_name}")


def _ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"rank": rank, **row} for rank, row in enumerate(rows, start=1)]


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
