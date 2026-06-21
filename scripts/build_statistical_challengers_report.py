#!/usr/bin/env python3
"""
Build the consolidated Phase 8A statistical challengers report.

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


REPORT_SOURCES = {
    "standalone_and_baselines": Path("reports/poisson_regression_comparison_first_run.json"),
    "dixon_coles_variants": Path("reports/dixon_coles_variants_first_run.json"),
    "time_decay_xg_variants": Path("reports/time_decay_xg_variants_first_run.json"),
}

OUTPUT_MD = Path("reports/statistical_challengers_phase_8a_summary.md")
OUTPUT_JSON = Path("reports/statistical_challengers_phase_8a_summary.json")

PRIMARY_MODELS = (
    "dc_fit_rho_each_fold",
    "txg_xg_pseudocount_010",
    "champion_dc_xg",
    "regularised_team_strength",
    "poisson_regression",
    "logistic_regression",
    "elo_baseline",
    "naive_outcome_rate",
)

SUPPORTING_VARIANTS = (
    "dc_rho_mild_minus_08",
    "dc_conservative_xg_shrinkage",
    "dc_score_grid_10",
    "dc_alpha_030",
    "dc_rho_stronger_minus_18",
    "txg_decay_90d",
    "txg_alpha_025",
    "txg_conservative_weighting",
    "txg_decay_45d",
)

MODEL_NOTES = {
    "dc_fit_rho_each_fold": "Best Phase 8A probability-quality candidate; fits rho inside each fold.",
    "txg_xg_pseudocount_010": "Most balanced champion-family candidate; improves Brier, log loss, and accuracy versus the original champion.",
    "champion_dc_xg": "Unchanged operational/reference model.",
    "regularised_team_strength": "Strongest standalone non-champion statistical challenger.",
    "poisson_regression": "Interpretable Poisson regression challenger; useful but below regularised team strength.",
    "logistic_regression": "Existing feature-based ML baseline carried forward to Phase 8B.",
    "elo_baseline": "Simple rating baseline.",
    "naive_outcome_rate": "Sanity-check baseline.",
}


def main() -> None:
    records = load_records()
    selected = [records[name] for name in PRIMARY_MODELS if name in records]
    supporting = [records[name] for name in SUPPORTING_VARIANTS if name in records]
    payload = build_payload(selected, supporting)
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_records() -> dict[str, dict[str, Any]]:
    """Load model comparison rows from first-run report JSON files."""
    records: dict[str, dict[str, Any]] = {}
    for source_name, source_path in REPORT_SOURCES.items():
        data = json.loads(source_path.read_text(encoding="utf-8"))
        for row in data["report_summary"]["comparison"]:
            model_name = row["model_name"]
            # Preserve the first configured source for each named model. The
            # source order intentionally takes the original champion comparison
            # row before later duplicate reference rows.
            records.setdefault(
                model_name,
                {
                    "model_name": model_name,
                    "n_matches": int(row["n_matches"]),
                    "brier_score": float(row["brier_score"]),
                    "log_loss": float(row["log_loss"]),
                    "accuracy": float(row["accuracy"]),
                    "source_report": source_path.as_posix(),
                    "source_group": source_name,
                    "note": MODEL_NOTES.get(model_name, ""),
                },
            )
    return records


def build_payload(selected: list[dict[str, Any]], supporting: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a serializable summary payload."""
    champion = _find(selected, "champion_dc_xg")
    dc_fit = _find(selected, "dc_fit_rho_each_fold")
    txg = _find(selected, "txg_xg_pseudocount_010")
    regularised = _find(selected, "regularised_team_strength")
    return {
        "title": "Phase 8A Consolidated Statistical Challengers Report",
        "source_reports": {name: path.as_posix() for name, path in REPORT_SOURCES.items()},
        "primary_models": selected,
        "supporting_variants": supporting,
        "rankings": {
            "brier_score": sorted(selected, key=lambda row: row["brier_score"]),
            "log_loss": sorted(selected, key=lambda row: row["log_loss"]),
            "accuracy": sorted(selected, key=lambda row: row["accuracy"], reverse=True),
        },
        "decision": {
            "operational_reference": champion,
            "best_probability_quality_candidate": dc_fit,
            "most_balanced_champion_family_candidate": txg,
            "strongest_standalone_challenger": regularised,
            "production_promotion": "none",
            "next_phase": "Phase 8B - improved logistic regression and feature-based ML challengers",
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the consolidated report."""
    decision = payload["decision"]
    lines = [
        "# Phase 8A Consolidated Statistical Challengers Report",
        "",
        "## Scope",
        "",
        "This report consolidates the Phase 8A first-run results for standalone statistical challengers,",
        "Dixon-Coles/champion-family configuration variants, and time-decay/xG-weighting variants.",
        "It is documentation/reporting only and does not change production prediction behaviour.",
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
        "## Supporting Phase 8A Variants",
        "",
        "These rows are useful context, but they are not the main carry-forward candidates.",
        "",
        _markdown_table(
            payload["supporting_variants"],
            ["model_name", "n_matches", "brier_score", "log_loss", "accuracy"],
        ),
        "",
        "## Interpretation",
        "",
        f"- Best probability-quality candidate: `{decision['best_probability_quality_candidate']['model_name']}` "
        f"with Brier {decision['best_probability_quality_candidate']['brier_score']:.4f} and "
        f"log loss {decision['best_probability_quality_candidate']['log_loss']:.4f}.",
        f"- Best accuracy candidate among primary rows: `{payload['rankings']['accuracy'][0]['model_name']}` "
        f"with accuracy {payload['rankings']['accuracy'][0]['accuracy']:.4f}.",
        f"- Strongest standalone non-champion challenger: `{decision['strongest_standalone_challenger']['model_name']}` "
        f"with Brier {decision['strongest_standalone_challenger']['brier_score']:.4f}, "
        f"log loss {decision['strongest_standalone_challenger']['log_loss']:.4f}, and "
        f"accuracy {decision['strongest_standalone_challenger']['accuracy']:.4f}.",
        "- `champion_dc_xg` remains the operational/reference model because the improvements are from one",
        "  evaluation window, some candidates trade probability quality against accuracy, and no candidate has",
        "  passed shadow/live-style validation yet.",
        "",
        "## Model Decision",
        "",
        "- No production promotion yet.",
        "- Carry `dc_fit_rho_each_fold` forward as the current best probability-quality candidate.",
        "- Carry `txg_xg_pseudocount_010` forward as the most balanced champion-family candidate because it",
        "  improves Brier score, log loss, and accuracy versus the original champion.",
        "- Keep `regularised_team_strength` as the strongest standalone non-champion statistical challenger.",
        "- Phase 8A is complete.",
        "- Next phase: Phase 8B - improved logistic regression and feature-based ML challengers.",
        "",
        "## Limitations",
        "",
        "- The main comparison covers one WSL season and 109 evaluated matches.",
        "- The WSL sample is small, so small differences can be unstable.",
        "- Champion-family variants were deliberately predeclared, but repeated testing still risks overfitting",
        "  to this evaluation window.",
        "- Candidate variants need shadow/live-style validation before any production decision.",
        "- Accuracy is a coarse metric and can disagree with probability-quality metrics such as Brier score and log loss.",
        "",
        "## Source Artifacts",
        "",
    ]
    for source_name, source_path in payload["source_reports"].items():
        lines.append(f"- `{source_name}`: `{source_path}`")
    lines.append("")
    return "\n".join(lines)


def _find(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    for row in rows:
        if row["model_name"] == model_name:
            return row
    raise ValueError(f"Missing expected model row: {model_name}")


def _ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked_rows = []
    for rank, row in enumerate(rows, start=1):
        ranked_rows.append({"rank": rank, **row})
    return ranked_rows


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
