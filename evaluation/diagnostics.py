"""
Champion diagnostics helpers for local model-comparison artefacts.

The functions in this module consume already-generated prediction rows. They do
not fit models, call Supabase, or alter prediction behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.compare import compare_model_results, comparison_to_records
from evaluation.failure_analysis import scored_prediction_rows

CHAMPION_MODEL = "champion_dc_xg"
MATCH_KEY_COLUMNS = ("fold_id", "match_date", "home_team", "away_team")


def load_model_comparison_payload(path: Path) -> dict[str, Any]:
    """Load a model-comparison JSON artefact and validate the expected row store."""
    if not path.exists():
        raise FileNotFoundError(f"Model comparison JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model comparison JSON must contain an object payload.")
    if "prediction_rows" not in payload:
        raise ValueError("Model comparison JSON is missing 'prediction_rows'.")
    if not isinstance(payload["prediction_rows"], list) or not payload["prediction_rows"]:
        raise ValueError("Model comparison JSON must contain at least one prediction row.")
    return payload


def build_champion_diagnostics(
    payload: dict[str, Any],
    *,
    champion_model: str = CHAMPION_MODEL,
    top_n: int = 10,
    high_confidence: float = 0.6,
) -> dict[str, Any]:
    """Build report-ready diagnostics for the champion model."""
    rows = pd.DataFrame(payload.get("prediction_rows", []))
    if rows.empty:
        raise ValueError("No prediction rows found in model comparison payload.")
    if "model_name" not in rows.columns:
        raise ValueError("Prediction rows must include 'model_name'.")

    champion_rows = rows[rows["model_name"] == champion_model].copy()
    if champion_rows.empty:
        raise ValueError(f"No prediction rows found for champion model '{champion_model}'.")

    champion_scored = scored_prediction_rows(champion_rows)
    comparison = compare_model_results(rows)
    comparison_records = comparison_to_records(comparison)
    champion_metrics = _find_model_record(comparison_records, champion_model)
    nearest_challenger = _nearest_challenger_record(comparison_records, champion_model)

    return {
        "champion_model": champion_model,
        "input_summary": {
            "n_prediction_rows": int(len(rows)),
            "n_champion_rows": int(len(champion_rows)),
            "models": sorted(rows["model_name"].dropna().astype(str).unique().tolist()),
        },
        "headline_metrics": champion_metrics,
        "nearest_challenger": nearest_challenger,
        "comparison": comparison_records,
        "high_confidence_misses": high_confidence_misses(
            champion_scored,
            n=top_n,
            min_confidence=high_confidence,
        ),
        "high_confidence_correct": high_confidence_correct(
            champion_scored,
            n=top_n,
            min_confidence=high_confidence,
        ),
        "confidence_bands": confidence_band_performance(champion_scored),
        "favourite_draw_breakdown": favourite_draw_breakdown(champion_scored),
        "team_error_summary": team_error_summary(champion_scored, n=top_n),
        "round_fold_summary": round_fold_summary(champion_scored),
        "challenger_comparison": champion_vs_challenger(rows, champion_model=champion_model),
        "limitations": [
            "Diagnostics use one generated comparison artefact only.",
            f"The champion sample contains {len(champion_rows)} evaluated matches.",
            "Rows are evaluated from completed fixtures already present in the comparison artefact.",
        ],
        "recommendations": recommendation_lines(champion_scored),
    }


def high_confidence_misses(
    scored_rows: pd.DataFrame | list[dict[str, Any]],
    *,
    n: int = 10,
    min_confidence: float = 0.6,
) -> list[dict[str, Any]]:
    """Return champion misses where the top predicted outcome was confident."""
    scored = _ensure_scored(scored_rows)
    misses = scored[(~scored["is_correct"]) & (scored["predicted_confidence"] >= min_confidence)]
    return _fixture_records(
        misses.sort_values(
            ["predicted_confidence", "row_log_loss"],
            ascending=[False, False],
            kind="mergesort",
        ).head(n)
    )


def high_confidence_correct(
    scored_rows: pd.DataFrame | list[dict[str, Any]],
    *,
    n: int = 10,
    min_confidence: float = 0.6,
) -> list[dict[str, Any]]:
    """Return champion correct predictions with the highest confidence."""
    scored = _ensure_scored(scored_rows)
    correct = scored[(scored["is_correct"]) & (scored["predicted_confidence"] >= min_confidence)]
    return _fixture_records(
        correct.sort_values(
            ["predicted_confidence", "actual_probability"],
            ascending=[False, False],
            kind="mergesort",
        ).head(n)
    )


def confidence_band_performance(scored_rows: pd.DataFrame | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarise champion performance by confidence bucket."""
    scored = _ensure_scored(scored_rows)
    grouped = (
        scored.groupby("confidence_bucket", sort=True)
        .agg(
            n=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
            mean_confidence=("predicted_confidence", "mean"),
            mean_actual_probability=("actual_probability", "mean"),
            mean_log_loss=("row_log_loss", "mean"),
        )
        .reset_index()
    )
    return _rounded_records(grouped)


def favourite_draw_breakdown(scored_rows: pd.DataFrame | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarise home favourites, away favourites, and predicted draw behaviour."""
    scored = _ensure_scored(scored_rows).copy()
    scored["favourite_type"] = scored["predicted_outcome"].map(
        {
            "H": "home_favourite",
            "D": "predicted_draw",
            "A": "away_favourite",
        }
    )
    grouped = (
        scored.groupby("favourite_type", sort=True)
        .agg(
            n=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
            mean_confidence=("predicted_confidence", "mean"),
            mean_actual_probability=("actual_probability", "mean"),
            mean_log_loss=("row_log_loss", "mean"),
        )
        .reset_index()
    )
    return _rounded_records(grouped)


def team_error_summary(
    scored_rows: pd.DataFrame | list[dict[str, Any]],
    *,
    n: int = 10,
) -> list[dict[str, Any]]:
    """Return team-level error rates across home and away appearances."""
    scored = _ensure_scored(scored_rows)
    team_rows = []
    for row in scored.itertuples(index=False):
        row_dict = row._asdict()
        for venue, team_column in (("home", "home_team"), ("away", "away_team")):
            team_rows.append(
                {
                    "team": row_dict.get(team_column),
                    "venue": venue,
                    "is_correct": row_dict["is_correct"],
                    "row_log_loss": row_dict["row_log_loss"],
                    "predicted_confidence": row_dict["predicted_confidence"],
                }
            )

    grouped = (
        pd.DataFrame(team_rows)
        .groupby("team", sort=True)
        .agg(
            n=("is_correct", "size"),
            error_rate=("is_correct", lambda values: 1.0 - float(values.mean())),
            mean_log_loss=("row_log_loss", "mean"),
            mean_confidence=("predicted_confidence", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(["error_rate", "mean_log_loss", "team"], ascending=[False, False, True])
    return _rounded_records(grouped.head(n))


def round_fold_summary(scored_rows: pd.DataFrame | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return fold or round-level instability summary."""
    scored = _ensure_scored(scored_rows)
    group_column = "round" if "round" in scored.columns else "fold_id"
    grouped = (
        scored.groupby(group_column, sort=True)
        .agg(
            n=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
            mean_confidence=("predicted_confidence", "mean"),
            mean_log_loss=("row_log_loss", "mean"),
        )
        .reset_index()
        .rename(columns={group_column: "round_or_fold"})
    )
    grouped = grouped.sort_values(["mean_log_loss", "round_or_fold"], ascending=[False, True])
    return _rounded_records(grouped)


def champion_vs_challenger(
    rows: pd.DataFrame | list[dict[str, Any]],
    *,
    champion_model: str = CHAMPION_MODEL,
) -> dict[str, Any] | None:
    """Compare champion row scoring against the nearest challenger where possible."""
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    comparison = compare_model_results(df)
    comparison_records = comparison_to_records(comparison)
    challenger = _nearest_challenger_record(comparison_records, champion_model)
    if challenger is None:
        return None

    challenger_name = challenger["model_name"]
    required = [column for column in MATCH_KEY_COLUMNS if column in df.columns]
    if not required:
        return {"challenger_model": challenger_name, "paired_matches": 0}

    champion = scored_prediction_rows(df[df["model_name"] == champion_model]).set_index(required)
    challenger_rows = scored_prediction_rows(df[df["model_name"] == challenger_name]).set_index(required)
    paired = champion.join(
        challenger_rows[["is_correct", "actual_probability", "row_log_loss"]],
        how="inner",
        lsuffix="_champion",
        rsuffix="_challenger",
    )
    if paired.empty:
        return {"challenger_model": challenger_name, "paired_matches": 0}

    return {
        "challenger_model": challenger_name,
        "paired_matches": int(len(paired)),
        "champion_only_correct": int((paired["is_correct_champion"] & ~paired["is_correct_challenger"]).sum()),
        "challenger_only_correct": int((~paired["is_correct_champion"] & paired["is_correct_challenger"]).sum()),
        "both_correct": int((paired["is_correct_champion"] & paired["is_correct_challenger"]).sum()),
        "both_wrong": int((~paired["is_correct_champion"] & ~paired["is_correct_challenger"]).sum()),
        "mean_log_loss_delta_champion_minus_challenger": round(
            float((paired["row_log_loss_champion"] - paired["row_log_loss_challenger"]).mean()),
            4,
        ),
        "mean_actual_probability_delta_champion_minus_challenger": round(
            float((paired["actual_probability_champion"] - paired["actual_probability_challenger"]).mean()),
            4,
        ),
    }


def recommendation_lines(scored_rows: pd.DataFrame | list[dict[str, Any]]) -> list[str]:
    """Generate conservative next-step recommendations from observed diagnostics."""
    scored = _ensure_scored(scored_rows)
    favourite_rows = favourite_draw_breakdown(scored)
    team_rows = team_error_summary(scored, n=3)
    recommendations = [
        "Keep champion_dc_xg as the reference model until a challenger beats it on the same folds.",
        "Test calibration improvements before adding new model families, because confidence gaps are visible in bands.",
    ]

    draw_row = next((row for row in favourite_rows if row["favourite_type"] == "predicted_draw"), None)
    if draw_row is None or draw_row["n"] == 0:
        recommendations.append("Add a draw-focused diagnostic/challenger feature test because the champion rarely predicts draws.")
    elif draw_row["accuracy"] < 0.5:
        recommendations.append("Investigate draw calibration because predicted draws underperform in this artefact.")

    if team_rows:
        recommendations.append(
            "Review team-specific residuals for "
            + ", ".join(str(row["team"]) for row in team_rows[:3])
            + " before changing the model form."
        )
    return recommendations


def render_markdown_report(summary: dict[str, Any]) -> str:
    """Render champion diagnostics as Markdown."""
    metrics = summary["headline_metrics"]
    challenger = summary.get("nearest_challenger")
    challenger_text = (
        f"Nearest challenger: {challenger['model_name']} "
        f"(Brier {challenger['brier_score']:.4f}, log loss {challenger['log_loss']:.4f}, "
        f"accuracy {challenger['accuracy']:.4f})."
        if challenger
        else "No challenger rows were available for comparison."
    )
    lines = [
        "# Champion Diagnostics Report",
        "",
        "## Executive Summary",
        "",
        (
            f"`{summary['champion_model']}` remains the reference model on "
            f"{summary['input_summary']['n_champion_rows']} evaluated matches: "
            f"Brier {metrics['brier_score']:.4f}, log loss {metrics['log_loss']:.4f}, "
            f"accuracy {metrics['accuracy']:.4f}."
        ),
        challenger_text,
        "This report diagnoses where the champion wins and fails; it does not change model behaviour.",
        "",
        "## Champion Headline Metrics",
        "",
        _markdown_table(
            [metrics],
            ["model_name", "n_matches", "brier_score", "log_loss", "accuracy", "rank"],
        ),
        "",
        "## Why Champion Remains The Reference Model",
        "",
        _markdown_table(
            summary["comparison"],
            ["rank", "model_name", "n_matches", "brier_score", "log_loss", "accuracy"],
        ),
        "",
        "## High-Confidence Misses",
        "",
        _markdown_table(summary["high_confidence_misses"], _fixture_columns()),
        "",
        "## High-Confidence Correct Predictions",
        "",
        _markdown_table(summary["high_confidence_correct"], _fixture_columns()),
        "",
        "## Confidence-Band Performance",
        "",
        _markdown_table(
            summary["confidence_bands"],
            ["confidence_bucket", "n", "accuracy", "mean_confidence", "mean_actual_probability", "mean_log_loss"],
        ),
        "",
        "## Draw And Favourite Behaviour",
        "",
        _markdown_table(
            summary["favourite_draw_breakdown"],
            ["favourite_type", "n", "accuracy", "mean_confidence", "mean_actual_probability", "mean_log_loss"],
        ),
        "",
        "## Team-Level Error Patterns",
        "",
        _markdown_table(
            summary["team_error_summary"],
            ["team", "n", "error_rate", "mean_log_loss", "mean_confidence"],
        ),
        "",
        "## Round/Fold Instability",
        "",
        _markdown_table(
            summary["round_fold_summary"],
            ["round_or_fold", "n", "accuracy", "mean_confidence", "mean_log_loss"],
        ),
        "",
        "## Champion Versus Nearest Challenger",
        "",
        _markdown_table(
            [summary["challenger_comparison"]] if summary["challenger_comparison"] else [],
            [
                "challenger_model",
                "paired_matches",
                "champion_only_correct",
                "challenger_only_correct",
                "both_correct",
                "both_wrong",
                "mean_log_loss_delta_champion_minus_challenger",
                "mean_actual_probability_delta_champion_minus_challenger",
            ],
        ),
        "",
        "## Limitations",
        "",
        *[f"- {line}" for line in summary["limitations"]],
        "",
        "## Specific Next Modelling Recommendations",
        "",
        *[f"- {line}" for line in summary["recommendations"]],
        "",
    ]
    return "\n".join(lines)


def _ensure_scored(rows: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if {"predicted_outcome", "predicted_confidence", "actual_probability", "row_log_loss", "is_correct"}.issubset(
        df.columns
    ):
        return df
    return scored_prediction_rows(df)


def _find_model_record(records: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    for record in records:
        if record["model_name"] == model_name:
            return record
    raise ValueError(f"No comparison metrics found for '{model_name}'.")


def _nearest_challenger_record(records: list[dict[str, Any]], champion_model: str) -> dict[str, Any] | None:
    for record in records:
        if record["model_name"] != champion_model:
            return record
    return None


def _fixture_records(rows: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [column for column in _fixture_columns() if column in rows.columns]
    return _rounded_records(rows.loc[:, columns])


def _fixture_columns() -> list[str]:
    return [
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


def _rounded_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, float):
                record[key] = round(value, 4)
    return records


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not columns:
        return "_No rows._"
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
