"""
Evaluation-only market-implied benchmark helpers.

The functions here treat odds as an external probability reference. They do not
feed market data into model training, feature generation or production routing.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.calibration import calibration_summary
from evaluation.compare import compare_model_results, comparison_to_records
from evaluation.failure_analysis import failure_analysis_summary, scored_prediction_rows
from evaluation.metrics import evaluate_prediction_set, validate_probabilities

RAW_COLUMNS = (
    "Date",
    "Home_Team",
    "Home_Goals",
    "Away_Goals",
    "Away_Team",
    "Odds_1",
    "Odds_X",
    "Odds_2",
    "Imp_Home",
    "Imp_Draw",
    "Imp_Away",
    "Overround",
    "P_Home",
    "P_Draw",
    "P_Away",
)
CANONICAL_COLUMNS = (
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "p_home_win",
    "p_draw",
    "p_away_win",
)
MODEL_NAME = "market_implied_benchmark"


def load_market_odds_csv(path: str | Path) -> pd.DataFrame:
    """Load the provided market odds CSV without mutating it."""
    return pd.read_csv(path)


def parse_fractional_or_decimal_odds(value: Any) -> float:
    """Return decimal odds from a fractional string or decimal value."""
    text = str(value).strip()
    if not text:
        raise ValueError("Odds value cannot be blank.")
    if "/" in text:
        return float(Fraction(text)) + 1.0
    decimal = float(text)
    if decimal <= 1.0:
        raise ValueError(f"Decimal odds must be greater than 1.0, got {value!r}.")
    return decimal


def normalise_market_rows(raw: pd.DataFrame, *, include_non_league: bool = False) -> pd.DataFrame:
    """Return canonical prediction-vs-result rows for market-only scoring."""
    missing = [column for column in RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Market odds CSV missing required columns: {missing}")

    df = raw.copy()
    if not include_non_league and "Note" in df.columns:
        df = df.loc[df["Note"].fillna("").astype(str).str.strip() == ""].copy()

    odds_frame = _derive_probabilities_from_odds(df)
    provided_probabilities = pd.DataFrame(
        {
            "provided_p_home": pd.to_numeric(df["P_Home"], errors="raise").to_numpy(dtype=float),
            "provided_p_draw": pd.to_numeric(df["P_Draw"], errors="raise").to_numpy(dtype=float),
            "provided_p_away": pd.to_numeric(df["P_Away"], errors="raise").to_numpy(dtype=float),
        },
        index=df.index,
    )
    max_abs_diff = (provided_probabilities.to_numpy(dtype=float) - odds_frame.loc[:, [
        "derived_p_home",
        "derived_p_draw",
        "derived_p_away",
    ]].to_numpy(dtype=float))
    max_abs_diff = np.abs(max_abs_diff).max(axis=1)

    normalized = pd.DataFrame(
        {
            "match_date": pd.to_datetime(df["Date"], format="ISO8601", errors="raise").dt.date.astype(str),
            "home_team": df["Home_Team"].astype(str).str.strip(),
            "away_team": df["Away_Team"].astype(str).str.strip(),
            "home_goals": pd.to_numeric(df["Home_Goals"], errors="raise").astype(int),
            "away_goals": pd.to_numeric(df["Away_Goals"], errors="raise").astype(int),
            "home_odds": df["Odds_1"].astype(str).str.strip(),
            "draw_odds": df["Odds_X"].astype(str).str.strip(),
            "away_odds": df["Odds_2"].astype(str).str.strip(),
            "decimal_home_odds": odds_frame["decimal_home_odds"].to_numpy(dtype=float),
            "decimal_draw_odds": odds_frame["decimal_draw_odds"].to_numpy(dtype=float),
            "decimal_away_odds": odds_frame["decimal_away_odds"].to_numpy(dtype=float),
            "raw_p_home": odds_frame["imp_home"].to_numpy(dtype=float),
            "raw_p_draw": odds_frame["imp_draw"].to_numpy(dtype=float),
            "raw_p_away": odds_frame["imp_away"].to_numpy(dtype=float),
            "provided_raw_p_home": pd.to_numeric(df["Imp_Home"], errors="raise"),
            "provided_raw_p_draw": pd.to_numeric(df["Imp_Draw"], errors="raise"),
            "provided_raw_p_away": pd.to_numeric(df["Imp_Away"], errors="raise"),
            "overround": odds_frame["overround"].to_numpy(dtype=float),
            "provided_overround": pd.to_numeric(df["Overround"], errors="raise"),
            "provided_p_home": provided_probabilities["provided_p_home"].to_numpy(dtype=float),
            "provided_p_draw": provided_probabilities["provided_p_draw"].to_numpy(dtype=float),
            "provided_p_away": provided_probabilities["provided_p_away"].to_numpy(dtype=float),
            "derived_p_home": odds_frame["derived_p_home"].to_numpy(dtype=float),
            "derived_p_draw": odds_frame["derived_p_draw"].to_numpy(dtype=float),
            "derived_p_away": odds_frame["derived_p_away"].to_numpy(dtype=float),
            "max_abs_provided_vs_derived_diff": max_abs_diff,
            "p_home_win": odds_frame["derived_p_home"].to_numpy(dtype=float),
            "p_draw": odds_frame["derived_p_draw"].to_numpy(dtype=float),
            "p_away_win": odds_frame["derived_p_away"].to_numpy(dtype=float),
            "note": df["Note"].fillna("").astype(str) if "Note" in df.columns else "",
            "model_name": MODEL_NAME,
        }
    )
    normalized["actual_outcome"] = [
        _actual_outcome(home, away)
        for home, away in normalized.loc[:, ["home_goals", "away_goals"]].itertuples(index=False)
    ]
    normalized["market_probability_source"] = "raw_odds_proportional_no_vig"
    normalized["odds_format"] = "fractional_or_decimal"
    return normalized.reset_index(drop=True)


def validate_market_rows(raw: pd.DataFrame, *, include_non_league: bool = False) -> dict[str, Any]:
    """Validate schema, probabilities, odds arithmetic and fixture quality."""
    missing = [column for column in RAW_COLUMNS if column not in raw.columns]
    if missing:
        return {
            "status": "error",
            "errors": [{"type": "missing_columns", "columns": missing}],
            "warnings": [],
        }

    full = normalise_market_rows(raw, include_non_league=True)
    evaluated = normalise_market_rows(raw, include_non_league=include_non_league)
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    _validate_required_values(evaluated, errors)
    _validate_fixture_duplicates(evaluated, warnings)
    _validate_probabilities(evaluated, errors, warnings)
    _validate_odds_quality(evaluated, errors, warnings)

    non_league_count = int((full["note"].astype(str).str.strip() != "").sum())
    if non_league_count and not include_non_league:
        warnings.append(
            {
                "type": "non_league_rows_excluded",
                "count": non_league_count,
                "notes": sorted(full.loc[full["note"].str.strip() != "", "note"].unique().tolist()),
            }
        )

    return {
        "status": "error" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "row_counts": {
            "raw_rows": int(len(raw)),
            "evaluated_rows": int(len(evaluated)),
            "excluded_non_league_rows": 0 if include_non_league else non_league_count,
        },
        "outcome_counts": evaluated["actual_outcome"].value_counts().sort_index().to_dict(),
        "date_range": {
            "min": str(evaluated["match_date"].min()) if not evaluated.empty else None,
            "max": str(evaluated["match_date"].max()) if not evaluated.empty else None,
        },
    }


def build_market_benchmark_result(
    raw: pd.DataFrame,
    *,
    include_non_league: bool = False,
    n_bins: int = 5,
    top_n: int = 10,
) -> dict[str, Any]:
    """Build market-only benchmark metrics, warnings and row-level scores."""
    validation = validate_market_rows(raw, include_non_league=include_non_league)
    if validation["errors"]:
        raise ValueError(f"Market benchmark validation failed: {validation['errors']}")

    rows = normalise_market_rows(raw, include_non_league=include_non_league)
    probabilities = rows.loc[:, ["p_home_win", "p_draw", "p_away_win"]].values.tolist()
    outcomes = rows["actual_outcome"].tolist()
    metrics = evaluate_prediction_set(probabilities, outcomes, n_bins=n_bins)
    comparison = compare_model_results(rows)
    scored_rows = scored_prediction_rows(rows)
    row_records = _row_level_records(scored_rows)
    return {
        "benchmark_type": "market_implied_probability_reference",
        "model_name": MODEL_NAME,
        "parameters": {
            "include_non_league": include_non_league,
            "n_bins": n_bins,
            "top_n": top_n,
            "probability_source": "raw odds converted to proportional no-vig probabilities",
            "raw_odds_columns": ["Odds_1", "Odds_X", "Odds_2"],
            "diagnostic_probability_columns": ["P_Home", "P_Draw", "P_Away"],
        },
        "validation": validation,
        "metrics": metrics,
        "comparison": comparison_to_records(comparison),
        "calibration": calibration_summary(probabilities, outcomes, n_bins=n_bins, min_bin_size=2),
        "failure_analysis": failure_analysis_summary(rows, n=top_n),
        "row_level_results": row_records,
    }


def render_market_benchmark_markdown(result: dict[str, Any]) -> str:
    """Render a concise market-only benchmark report."""
    validation = result["validation"]
    metrics = result["metrics"]
    lines = [
        "# WSL Market-Implied Benchmark",
        "",
        (
            "This report derives proportional no-vig probabilities directly from the raw fractional odds "
            "columns (`Odds_1`, `Odds_X`, `Odds_2`) and evaluates them as an external market probability "
            "reference. The supplied de-vigged probability columns are used only for data-quality "
            "diagnostics. This layer is evaluation-only and is not used for model training or features."
        ),
        "",
        "## Summary",
        "",
        f"- Benchmark: `{result['model_name']}`",
        f"- Fixture count: {metrics['n_matches']}",
        f"- Brier score: {metrics['brier_score']:.6f}",
        f"- Log loss: {metrics['log_loss']:.6f}",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Raw rows: {validation['row_counts']['raw_rows']}",
        f"- Evaluated rows: {validation['row_counts']['evaluated_rows']}",
        f"- Excluded non-league rows: {validation['row_counts']['excluded_non_league_rows']}",
        "",
        "## Data Quality Warnings",
        "",
        _markdown_table(validation["warnings"], _warning_columns(validation["warnings"])),
        "",
        "## Metric Table",
        "",
        _markdown_table(
            result["comparison"],
            ["rank", "model_name", "n_matches", "brier_score", "log_loss", "accuracy"],
        ),
        "",
        "## Calibration",
        "",
        _markdown_table(
            result["calibration"]["calibration_bins"],
            ["bin", "count", "mean_confidence", "observed_accuracy", "calibration_gap", "is_sparse"],
        ),
        "",
        "## Confidence Buckets",
        "",
        _markdown_table(
            result["calibration"]["confidence_buckets"],
            ["bucket", "count", "mean_confidence", "accuracy", "calibration_gap", "is_sparse"],
        ),
        "",
        "## Worst Misses",
        "",
        _markdown_table(
            result["failure_analysis"]["worst_misses"],
            [
                "match_date",
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
        "## Notes",
        "",
        "- Do not interpret this as a production decision artifact.",
        "- Final published conclusions require verified odds source, snapshot timing and licensing.",
    ]
    return "\n".join(lines) + "\n"


def write_market_outputs(
    result: dict[str, Any],
    *,
    output_md: str | Path | None = None,
    output_json: str | Path | None = None,
    output_rows: str | Path | None = None,
) -> None:
    """Write optional Markdown, JSON and row-level CSV artefacts."""
    if output_md:
        Path(output_md).write_text(render_market_benchmark_markdown(result), encoding="utf-8")
    if output_json:
        json_payload = {key: value for key, value in result.items() if key != "row_level_results"}
        Path(output_json).write_text(
            json.dumps(json_payload, default=_json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if output_rows:
        pd.DataFrame(result["row_level_results"]).to_csv(output_rows, index=False)


def _validate_required_values(df: pd.DataFrame, errors: list[dict[str, Any]]) -> None:
    for column in (*CANONICAL_COLUMNS, "home_odds", "draw_odds", "away_odds", "raw_p_home", "raw_p_draw", "raw_p_away"):
        missing = int(df[column].isna().sum()) if column in df.columns else len(df)
        blanks = int((df[column].astype(str).str.strip() == "").sum()) if column in df.columns else len(df)
        count = max(missing, blanks)
        if count:
            errors.append({"type": "missing_required_values", "column": column, "count": count})


def _validate_fixture_duplicates(df: pd.DataFrame, warnings: list[dict[str, Any]]) -> None:
    duplicated = df.duplicated(["match_date", "home_team", "away_team"], keep=False)
    if duplicated.any():
        warnings.append(
            {
                "type": "duplicate_fixtures",
                "count": int(duplicated.sum()),
                "examples": df.loc[duplicated, ["match_date", "home_team", "away_team"]]
                .head(10)
                .to_dict(orient="records"),
            }
        )


def _validate_probabilities(df: pd.DataFrame, errors: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    probability_columns = ["p_home_win", "p_draw", "p_away_win"]
    probabilities = df.loc[:, probability_columns].astype(float)
    if (probabilities < 0).any().any() or (probabilities > 1).any().any():
        errors.append({"type": "probability_out_of_range"})
    try:
        validate_probabilities(probabilities.values.tolist(), normalize=False)
    except ValueError:
        row_sums = probabilities.sum(axis=1)
        warnings.append(
            {
                "type": "probability_sum_drift",
                "max_abs_drift": round(float((row_sums - 1.0).abs().max()), 8),
                "rows_over_1e_4": int(((row_sums - 1.0).abs() > 1e-4).sum()),
            }
        )


def _validate_odds_quality(df: pd.DataFrame, errors: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    rows = []
    for idx, row in df.iterrows():
        provided_implied = np.asarray(
            [row["provided_raw_p_home"], row["provided_raw_p_draw"], row["provided_raw_p_away"]],
            dtype=float,
        )
        derived_implied = np.asarray([row["raw_p_home"], row["raw_p_draw"], row["raw_p_away"]], dtype=float)
        derived_fair = np.asarray([row["derived_p_home"], row["derived_p_draw"], row["derived_p_away"]], dtype=float)
        provided_fair = np.asarray(
            [row["provided_p_home"], row["provided_p_draw"], row["provided_p_away"]],
            dtype=float,
        )
        rows.append(
            {
                "idx": int(idx),
                "match_date": row["match_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "implied_max_diff": float(np.abs(provided_implied - derived_implied).max()),
                "overround_diff": abs(float(row["provided_overround"]) - float(row["overround"])),
                "max_abs_provided_vs_derived_diff": float(np.abs(provided_fair - derived_fair).max()),
                "overround": float(row["overround"]),
                "provided_p_home": float(row["provided_p_home"]),
                "provided_p_draw": float(row["provided_p_draw"]),
                "provided_p_away": float(row["provided_p_away"]),
                "derived_p_home": float(row["derived_p_home"]),
                "derived_p_draw": float(row["derived_p_draw"]),
                "derived_p_away": float(row["derived_p_away"]),
            }
        )

    if not rows:
        return

    implied_bad = [item for item in rows if item["implied_max_diff"] > 5e-4]
    overround_bad = [item for item in rows if item["overround_diff"] > 5e-4]
    underround_rows = [item for item in rows if item["overround"] < 1.0]
    provided_vs_derived_bad = [item for item in rows if item["max_abs_provided_vs_derived_diff"] > 0.05]

    if implied_bad:
        warnings.append(
            {
                "type": "implied_probability_mismatch",
                "count": len(implied_bad),
                "examples": implied_bad[:10],
            }
        )
    if overround_bad:
        warnings.append({"type": "overround_mismatch", "count": len(overround_bad), "examples": overround_bad[:10]})
    if underround_rows:
        warnings.append({"type": "underround_rows", "count": len(underround_rows), "examples": underround_rows[:10]})
    if provided_vs_derived_bad:
        warnings.append(
            {
                "type": "provided_vs_derived_probability_mismatch",
                "count": len(provided_vs_derived_bad),
                "threshold": 0.05,
                "examples": provided_vs_derived_bad[:10],
            }
        )

    max_diff = max(item["max_abs_provided_vs_derived_diff"] for item in rows)
    warnings.append(
        {
            "type": "provided_probability_reference_check",
            "message": (
                "Benchmark probabilities are derived from raw odds. Supplied P_Home/P_Draw/P_Away values "
                "are retained only as diagnostics."
            ),
            "max_abs_provided_vs_derived_diff": round(max_diff, 6),
        }
    )


def _row_level_records(scored_rows: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "actual_outcome",
        "predicted_outcome",
        "predicted_confidence",
        "actual_probability",
        "row_log_loss",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "derived_p_home",
        "derived_p_draw",
        "derived_p_away",
        "provided_p_home",
        "provided_p_draw",
        "provided_p_away",
        "max_abs_provided_vs_derived_diff",
        "home_odds",
        "draw_odds",
        "away_odds",
        "overround",
        "note",
    ]
    records = scored_rows.loc[:, [column for column in columns if column in scored_rows.columns]].to_dict(
        orient="records"
    )
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, float) and math.isfinite(value):
                record[key] = round(value, 6)
    return records


def _actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _warning_columns(warnings: list[dict[str, Any]]) -> list[str]:
    if not warnings:
        return ["type", "count", "message"]
    preferred = [
        "type",
        "count",
        "threshold",
        "max_abs_provided_vs_derived_diff",
        "max_abs_drift",
        "rows_over_1e_4",
        "message",
    ]
    present = {key for row in warnings for key in row}
    return [column for column in preferred if column in present]


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not columns:
        return "_None_"
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
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _derive_probabilities_from_odds(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for row in df.loc[:, ["Odds_1", "Odds_X", "Odds_2"]].itertuples(index=False):
        decimal_home = parse_fractional_or_decimal_odds(row.Odds_1)
        decimal_draw = parse_fractional_or_decimal_odds(row.Odds_X)
        decimal_away = parse_fractional_or_decimal_odds(row.Odds_2)
        imp_home = 1.0 / decimal_home
        imp_draw = 1.0 / decimal_draw
        imp_away = 1.0 / decimal_away
        overround = imp_home + imp_draw + imp_away
        if overround <= 0:
            raise ValueError("Odds-derived overround must be positive.")
        records.append(
            {
                "decimal_home_odds": decimal_home,
                "decimal_draw_odds": decimal_draw,
                "decimal_away_odds": decimal_away,
                "imp_home": imp_home,
                "imp_draw": imp_draw,
                "imp_away": imp_away,
                "overround": overround,
                "derived_p_home": imp_home / overround,
                "derived_p_draw": imp_draw / overround,
                "derived_p_away": imp_away / overround,
            }
        )
    return pd.DataFrame(records, index=df.index)
