"""
Offline model-vs-market comparison helpers.

This module joins existing model prediction artefacts to an external
market-implied probability reference on matched fixtures only. Market odds stay
evaluation-only: they are not used as model features, training inputs or
production prediction inputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.blending import normalise_probability_frame
from evaluation.calibration import calibration_summary
from evaluation.compare import compare_model_results, comparison_to_records
from evaluation.market_benchmark import MODEL_NAME as MARKET_MODEL_NAME
from evaluation.market_benchmark import load_market_odds_csv, normalise_market_rows
from evaluation.metrics import OUTCOME_LABELS, brier_score_3way, multiclass_log_loss, outcome_accuracy

TEAM_ALIASES = {
    "manchester united": "manchester utd",
    "tottenham hotspur": "tottenham",
    "leicester city": "leicester",
    "west ham united": "west ham",
}

PROBABILITY_COLUMNS = ("p_home_win", "p_draw", "p_away_win")
KEY_COLUMNS = ("match_date_key", "normalized_home_team", "normalized_away_team")


def load_model_prediction_rows(path: str | Path) -> pd.DataFrame:
    """Load row-level model predictions from a JSON or CSV artefact."""
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("prediction_rows") if isinstance(payload, dict) else payload
    if rows is None:
        raise ValueError("Model prediction artefact must contain a prediction_rows list.")
    return pd.DataFrame(rows)


def prepare_model_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalize model rows for matched-fixture comparison."""
    required = {"match_date", "home_team", "away_team", "model_name", *PROBABILITY_COLUMNS}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Model rows missing required columns: {missing}")

    prepared = normalise_probability_frame(rows).copy()
    prepared = _add_fixture_keys(prepared)
    if "actual_outcome" not in prepared.columns:
        if {"home_goals", "away_goals"}.issubset(prepared.columns):
            prepared["actual_outcome"] = [
                _outcome_from_goals(home, away)
                for home, away in prepared.loc[:, ["home_goals", "away_goals"]].itertuples(index=False)
            ]
        else:
            raise ValueError("Model rows must include actual_outcome or home_goals/away_goals.")
    prepared["actual_outcome"] = prepared["actual_outcome"].map(_normalise_outcome)
    return prepared


def prepare_market_rows(raw_market_rows: pd.DataFrame) -> pd.DataFrame:
    """Build odds-derived market rows and normalize fixture keys."""
    market = normalise_market_rows(raw_market_rows)
    market = _add_fixture_keys(market)
    return market


def normalize_team_name(value: Any) -> str:
    """Normalize team names for fallback fixture matching."""
    normalized = str(value).strip().casefold()
    return TEAM_ALIASES.get(normalized, normalized)


def normalize_fixture_date(value: Any) -> str:
    """Return YYYY-MM-DD date strings for fixture matching."""
    return pd.to_datetime(value, format="ISO8601", errors="raise").date().isoformat()


def match_model_to_market(
    model_rows: pd.DataFrame,
    market_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Join model rows to market rows on normalized date/home/away keys."""
    model = prepare_model_rows(model_rows)
    market = prepare_market_rows(market_rows)

    model_keys = _unique_fixture_records(model)
    market_keys = _unique_fixture_records(market)
    model_key_set = {row["fixture_key"] for row in model_keys}
    market_key_set = {row["fixture_key"] for row in market_keys}
    unmatched_model = [row for row in model_keys if row["fixture_key"] not in market_key_set]
    unmatched_market = [row for row in market_keys if row["fixture_key"] not in model_key_set]

    market_prefixed = _market_join_columns(market)
    joined = model.merge(market_prefixed, how="inner", on=list(KEY_COLUMNS), suffixes=("", "_market_join"))
    return joined, unmatched_model, unmatched_market


def build_model_market_comparison(
    model_rows: pd.DataFrame,
    market_rows: pd.DataFrame,
    *,
    n_bins: int = 5,
    top_n: int = 10,
) -> dict[str, Any]:
    """Build matched model-vs-market metrics and report sections."""
    joined, unmatched_model, unmatched_market = match_model_to_market(model_rows, market_rows)
    if joined.empty:
        raise ValueError("No model rows matched market rows. Check fixture dates and team aliases.")

    row_level = build_row_level_comparison(joined)
    combined_rows = _combined_metric_rows(row_level)
    metric_table = build_metric_table(combined_rows)
    calibration = build_calibration_comparison(combined_rows, n_bins=n_bins)
    disagreement = build_disagreement_analysis(row_level, top_n=top_n)
    draw_sensitivity = build_draw_sensitivity(row_level)
    underdog = build_market_favourite_analysis(row_level, top_n=top_n)
    worst_deltas = build_worst_delta_analysis(row_level, top_n=top_n)

    matched_fixture_count = int(row_level["fixture_key"].nunique())
    return {
        "comparison_type": "matched_fixture_model_vs_market",
        "parameters": {
            "n_bins": n_bins,
            "top_n": top_n,
            "fixture_key": list(KEY_COLUMNS),
            "team_aliases": TEAM_ALIASES,
            "market_probability_source": "raw odds converted to proportional no-vig probabilities",
        },
        "data_snapshot": {
            "model_fixture_count": int(prepare_model_rows(model_rows)["fixture_key"].nunique()),
            "market_fixture_count": int(prepare_market_rows(market_rows)["fixture_key"].nunique()),
            "matched_fixture_count": matched_fixture_count,
            "matched_row_count": int(len(row_level)),
            "unmatched_model_fixture_count": len(unmatched_model),
            "unmatched_market_fixture_count": len(unmatched_market),
            "unmatched_model_fixtures": unmatched_model,
            "unmatched_market_fixtures": unmatched_market,
        },
        "metrics": metric_table,
        "calibration": calibration,
        "disagreement_analysis": disagreement,
        "draw_sensitivity": draw_sensitivity,
        "market_favourite_analysis": underdog,
        "worst_deltas": worst_deltas,
        "row_level_results": row_level.to_dict(orient="records"),
    }


def build_row_level_comparison(joined_rows: pd.DataFrame) -> pd.DataFrame:
    """Return one model-vs-market comparison row per matched fixture/model."""
    records: list[dict[str, Any]] = []
    for row in joined_rows.itertuples(index=False):
        model_probs = np.asarray(
            [row.p_home_win, row.p_draw, row.p_away_win],
            dtype=float,
        )
        market_probs = np.asarray(
            [row.market_p_home_win, row.market_p_draw, row.market_p_away_win],
            dtype=float,
        )
        actual = str(row.actual_outcome)
        actual_idx = OUTCOME_LABELS.index(actual)
        model_pick_idx = int(model_probs.argmax())
        market_pick_idx = int(market_probs.argmax())
        model_actual_probability = float(model_probs[actual_idx])
        market_actual_probability = float(market_probs[actual_idx])
        model_row_log_loss = _row_log_loss(model_actual_probability)
        market_row_log_loss = _row_log_loss(market_actual_probability)
        market_favourite_idx = market_pick_idx
        market_underdog_idx = int(market_probs.argmin())
        records.append(
            {
                "fixture_key": row.fixture_key,
                "match_date": row.match_date_key,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_goals": int(row.home_goals),
                "away_goals": int(row.away_goals),
                "actual_outcome": actual,
                "model_name": row.model_name,
                "model_p_home_win": float(model_probs[0]),
                "model_p_draw": float(model_probs[1]),
                "model_p_away_win": float(model_probs[2]),
                "model_pick": OUTCOME_LABELS[model_pick_idx],
                "model_actual_probability": model_actual_probability,
                "model_row_log_loss": model_row_log_loss,
                "market_p_home_win": float(market_probs[0]),
                "market_p_draw": float(market_probs[1]),
                "market_p_away_win": float(market_probs[2]),
                "market_pick": OUTCOME_LABELS[market_pick_idx],
                "market_actual_probability": market_actual_probability,
                "market_row_log_loss": market_row_log_loss,
                "actual_probability_delta_model_minus_market": model_actual_probability - market_actual_probability,
                "log_loss_delta_model_minus_market": model_row_log_loss - market_row_log_loss,
                "max_probability_gap": float(np.abs(model_probs - market_probs).max()),
                "pick_disagrees": model_pick_idx != market_pick_idx,
                "market_favourite": OUTCOME_LABELS[market_favourite_idx],
                "market_favourite_probability": float(market_probs[market_favourite_idx]),
                "market_favourite_won": OUTCOME_LABELS[market_favourite_idx] == actual,
                "market_underdog": OUTCOME_LABELS[market_underdog_idx],
                "market_underdog_probability": float(market_probs[market_underdog_idx]),
                "market_underdog_won": OUTCOME_LABELS[market_underdog_idx] == actual,
                "model_market_favourite_probability": float(model_probs[market_favourite_idx]),
                "model_less_confident_in_market_favourite": (
                    float(model_probs[market_favourite_idx]) < float(market_probs[market_favourite_idx])
                ),
                "home_odds": row.home_odds,
                "draw_odds": row.draw_odds,
                "away_odds": row.away_odds,
                "overround": float(row.overround),
                "provided_p_home": float(row.provided_p_home),
                "provided_p_draw": float(row.provided_p_draw),
                "provided_p_away": float(row.provided_p_away),
                "max_abs_provided_vs_derived_diff": float(row.max_abs_provided_vs_derived_diff),
            }
        )
    result = pd.DataFrame(records)
    return _round_float_columns(result)


def build_metric_table(combined_rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Return report metrics, including mean actual-outcome probability."""
    comparison = compare_model_results(combined_rows)
    actual_probability = (
        combined_rows.assign(
            actual_probability=[
                _actual_probability(row)
                for row in combined_rows.loc[:, ["p_home_win", "p_draw", "p_away_win", "actual_outcome"]].to_dict(
                    orient="records"
                )
            ]
        )
        .groupby("model_name")["actual_probability"]
        .mean()
        .to_dict()
    )
    comparison["avg_actual_probability"] = comparison["model_name"].map(actual_probability)
    columns = ["rank", "model_name", "n_matches", "brier_score", "log_loss", "accuracy", "avg_actual_probability"]
    return _round_records(comparison.loc[:, columns].to_dict(orient="records"))


def build_calibration_comparison(combined_rows: pd.DataFrame, *, n_bins: int = 5) -> dict[str, Any]:
    """Return calibration summaries by model/market row set."""
    summaries: dict[str, Any] = {}
    for model_name, rows in combined_rows.groupby("model_name", sort=True):
        probabilities = rows.loc[:, PROBABILITY_COLUMNS].astype(float).values.tolist()
        outcomes = rows["actual_outcome"].tolist()
        summaries[str(model_name)] = calibration_summary(probabilities, outcomes, n_bins=n_bins, min_bin_size=2)
    return summaries


def build_disagreement_analysis(row_level: pd.DataFrame, *, top_n: int = 10) -> dict[str, Any]:
    """Summarize model pick disagreements against the market-implied pick."""
    rows = row_level.copy()
    rows["model_hit"] = rows["model_pick"] == rows["actual_outcome"]
    rows["market_hit"] = rows["market_pick"] == rows["actual_outcome"]
    disagreed = rows[rows["pick_disagrees"]].copy()
    summary = (
        disagreed.groupby("model_name", sort=True)
        .agg(
            disagreement_count=("pick_disagrees", "size"),
            model_hit_rate_when_disagreeing=("model_hit", "mean"),
            market_hit_rate_when_disagreeing=("market_hit", "mean"),
            mean_max_probability_gap=("max_probability_gap", "mean"),
        )
        .reset_index()
    )
    all_models = pd.DataFrame({"model_name": sorted(rows["model_name"].unique())})
    summary = all_models.merge(summary, how="left", on="model_name").fillna(0)
    top = disagreed.sort_values(["max_probability_gap", "model_name"], ascending=[False, True], kind="mergesort").head(
        top_n
    )
    return {
        "summary": _round_records(summary.to_dict(orient="records")),
        "top_disagreements": _round_records(_display_rows(top).to_dict(orient="records")),
    }


def build_draw_sensitivity(row_level: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize draw probability and draw-pick behaviour by model."""
    records = []
    for model_name, rows in row_level.groupby("model_name", sort=True):
        actual_draws = rows[rows["actual_outcome"] == "D"]
        records.append(
            {
                "model_name": model_name,
                "n_matches": int(len(rows)),
                "actual_draw_count": int(len(actual_draws)),
                "mean_model_draw_probability": float(rows["model_p_draw"].mean()),
                "mean_market_draw_probability": float(rows["market_p_draw"].mean()),
                "model_draw_pick_count": int((rows["model_pick"] == "D").sum()),
                "market_draw_pick_count": int((rows["market_pick"] == "D").sum()),
                "model_actual_draw_probability": (
                    None if actual_draws.empty else float(actual_draws["model_p_draw"].mean())
                ),
                "market_actual_draw_probability": (
                    None if actual_draws.empty else float(actual_draws["market_p_draw"].mean())
                ),
            }
        )
    return _round_records(records)


def build_market_favourite_analysis(row_level: pd.DataFrame, *, top_n: int = 10) -> dict[str, Any]:
    """Summarize market-favourite outcomes and underdog results."""
    fixture_rows = row_level.drop_duplicates("fixture_key").copy()
    summary = {
        "n_fixtures": int(len(fixture_rows)),
        "market_favourite_wins": int(fixture_rows["market_favourite_won"].sum()),
        "market_favourite_fails_to_win": int((~fixture_rows["market_favourite_won"]).sum()),
        "market_underdog_wins": int(fixture_rows["market_underdog_won"].sum()),
    }
    failed_less_confident = row_level[
        (~row_level["market_favourite_won"]) & (row_level["model_less_confident_in_market_favourite"])
    ].copy()
    failed_less_confident["market_favourite_probability_gap"] = (
        failed_less_confident["market_favourite_probability"]
        - failed_less_confident["model_market_favourite_probability"]
    )
    cases = failed_less_confident.sort_values(
        ["market_favourite_probability_gap", "model_name"],
        ascending=[False, True],
        kind="mergesort",
    ).head(top_n)
    return {
        "summary": summary,
        "model_less_confident_in_failed_market_favourite": _round_records(
            _display_rows(cases).to_dict(orient="records")
        ),
    }


def build_worst_delta_analysis(row_level: pd.DataFrame, *, top_n: int = 10) -> dict[str, Any]:
    """Return largest row-log-loss deltas in both directions."""
    model_better = row_level.sort_values(
        ["log_loss_delta_model_minus_market", "model_name"],
        ascending=[True, True],
        kind="mergesort",
    ).head(top_n)
    market_better = row_level.sort_values(
        ["log_loss_delta_model_minus_market", "model_name"],
        ascending=[False, True],
        kind="mergesort",
    ).head(top_n)
    return {
        "model_outperformed_market_by_log_loss": _round_records(_display_rows(model_better).to_dict(orient="records")),
        "market_outperformed_model_by_log_loss": _round_records(_display_rows(market_better).to_dict(orient="records")),
    }


def render_model_market_markdown(result: dict[str, Any]) -> str:
    """Render a concise matched-fixture model-vs-market report."""
    snapshot = result["data_snapshot"]
    lines = [
        "# WSL Model vs Market-Implied Benchmark",
        "",
        (
            "This is an evaluation-only matched-fixture comparison between existing model prediction "
            "artefacts and an external market probability reference. Market probabilities are derived "
            "from raw odds using proportional no-vig normalization and are not used as model features, "
            "training data or production prediction inputs."
        ),
        "",
        "## Summary",
        "",
        f"- Model fixtures: {snapshot['model_fixture_count']}",
        f"- Market fixtures: {snapshot['market_fixture_count']}",
        f"- Matched fixtures: {snapshot['matched_fixture_count']}",
        f"- Matched model rows: {snapshot['matched_row_count']}",
        f"- Unmatched model fixtures: {snapshot['unmatched_model_fixture_count']}",
        f"- Unmatched market fixtures: {snapshot['unmatched_market_fixture_count']}",
        "",
        "## Metric Table",
        "",
        _markdown_table(
            result["metrics"],
            ["rank", "model_name", "n_matches", "brier_score", "log_loss", "accuracy", "avg_actual_probability"],
        ),
        "",
        "## Disagreement Analysis",
        "",
        _markdown_table(
            result["disagreement_analysis"]["summary"],
            [
                "model_name",
                "disagreement_count",
                "model_hit_rate_when_disagreeing",
                "market_hit_rate_when_disagreeing",
                "mean_max_probability_gap",
            ],
        ),
        "",
        "## Top Disagreements",
        "",
        _markdown_table(result["disagreement_analysis"]["top_disagreements"], _analysis_columns()),
        "",
        "## Draw Sensitivity",
        "",
        _markdown_table(
            result["draw_sensitivity"],
            [
                "model_name",
                "actual_draw_count",
                "mean_model_draw_probability",
                "mean_market_draw_probability",
                "model_draw_pick_count",
                "market_draw_pick_count",
                "model_actual_draw_probability",
                "market_actual_draw_probability",
            ],
        ),
        "",
        "## Market-Favourite Failure Analysis",
        "",
        _markdown_table(
            [result["market_favourite_analysis"]["summary"]],
            list(result["market_favourite_analysis"]["summary"]),
        ),
        "",
        "## Model Less Confident In Failed Market Favourites",
        "",
        _markdown_table(
            result["market_favourite_analysis"]["model_less_confident_in_failed_market_favourite"],
            _analysis_columns(extra=["market_favourite_probability_gap"]),
        ),
        "",
        "## Model Outperformed Market By Row Log Loss",
        "",
        _markdown_table(result["worst_deltas"]["model_outperformed_market_by_log_loss"], _analysis_columns()),
        "",
        "## Market Outperformed Model By Row Log Loss",
        "",
        _markdown_table(result["worst_deltas"]["market_outperformed_model_by_log_loss"], _analysis_columns()),
        "",
        "## Unmatched Fixtures",
        "",
        f"- Unmatched model fixtures: {snapshot['unmatched_model_fixture_count']}",
        f"- Unmatched market fixtures: {snapshot['unmatched_market_fixture_count']}",
        "",
        "## Guardrails",
        "",
        "- This is a matched-fixture comparison for evaluation only.",
        "- Market odds are not used as model features or training data.",
        "- This is not a market blending implementation.",
        "- Interpret results only with source, timing and licensing limitations in mind.",
    ]
    return "\n".join(lines) + "\n"


def write_model_market_outputs(
    result: dict[str, Any],
    *,
    output_md: str | Path | None = None,
    output_json: str | Path | None = None,
    output_rows: str | Path | None = None,
) -> None:
    """Write optional Markdown, JSON and row-level CSV artefacts."""
    if output_md:
        Path(output_md).write_text(render_model_market_markdown(result), encoding="utf-8")
    if output_json:
        payload = {key: value for key, value in result.items() if key != "row_level_results"}
        Path(output_json).write_text(
            json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if output_rows:
        pd.DataFrame(result["row_level_results"]).to_csv(output_rows, index=False)


def _add_fixture_keys(rows: pd.DataFrame) -> pd.DataFrame:
    prepared = rows.copy()
    prepared["match_date_key"] = prepared["match_date"].map(normalize_fixture_date)
    prepared["normalized_home_team"] = prepared["home_team"].map(normalize_team_name)
    prepared["normalized_away_team"] = prepared["away_team"].map(normalize_team_name)
    prepared["fixture_key"] = [
        "|".join(items)
        for items in prepared.loc[:, ["match_date_key", "normalized_home_team", "normalized_away_team"]].itertuples(
            index=False,
            name=None,
        )
    ]
    return prepared


def _unique_fixture_records(rows: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "fixture_key",
        "match_date_key",
        "home_team",
        "away_team",
        "normalized_home_team",
        "normalized_away_team",
    ]
    return (
        rows.loc[:, columns]
        .drop_duplicates("fixture_key")
        .sort_values(["match_date_key", "normalized_home_team", "normalized_away_team"], kind="mergesort")
        .to_dict(orient="records")
    )


def _market_join_columns(market: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *KEY_COLUMNS,
        "market_probability_source",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "home_odds",
        "draw_odds",
        "away_odds",
        "overround",
        "provided_p_home",
        "provided_p_draw",
        "provided_p_away",
        "max_abs_provided_vs_derived_diff",
    ]
    selected = market.loc[:, columns].drop_duplicates(list(KEY_COLUMNS)).copy()
    return selected.rename(
        columns={
            "p_home_win": "market_p_home_win",
            "p_draw": "market_p_draw",
            "p_away_win": "market_p_away_win",
        }
    )


def _combined_metric_rows(row_level: pd.DataFrame) -> pd.DataFrame:
    model_rows = row_level.loc[
        :,
        [
            "fixture_key",
            "match_date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "actual_outcome",
            "model_name",
            "model_p_home_win",
            "model_p_draw",
            "model_p_away_win",
        ],
    ].rename(
        columns={
            "model_p_home_win": "p_home_win",
            "model_p_draw": "p_draw",
            "model_p_away_win": "p_away_win",
        }
    )
    market_rows = row_level.drop_duplicates("fixture_key").loc[
        :,
        [
            "fixture_key",
            "match_date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "actual_outcome",
            "market_p_home_win",
            "market_p_draw",
            "market_p_away_win",
        ],
    ].rename(
        columns={
            "market_p_home_win": "p_home_win",
            "market_p_draw": "p_draw",
            "market_p_away_win": "p_away_win",
        }
    )
    market_rows["model_name"] = MARKET_MODEL_NAME
    return pd.concat([model_rows, market_rows], ignore_index=True)


def _actual_probability(row: dict[str, Any]) -> float:
    label = str(row["actual_outcome"])
    index = OUTCOME_LABELS.index(label)
    return float([row["p_home_win"], row["p_draw"], row["p_away_win"]][index])


def _row_log_loss(probability: float) -> float:
    return float(-math.log(min(max(probability, 1e-15), 1.0)))


def _outcome_from_goals(home_goals: Any, away_goals: Any) -> str:
    home = int(home_goals)
    away = int(away_goals)
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"


def _normalise_outcome(value: Any) -> str:
    label = str(value).strip().upper()
    aliases = {"HOME": "H", "HOME_WIN": "H", "H": "H", "DRAW": "D", "D": "D", "AWAY": "A", "AWAY_WIN": "A", "A": "A"}
    if label not in aliases:
        raise ValueError(f"Invalid actual outcome: {value!r}. Expected H, D or A.")
    return aliases[label]


def _display_rows(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_name",
        "match_date",
        "home_team",
        "away_team",
        "actual_outcome",
        "model_pick",
        "market_pick",
        "model_actual_probability",
        "market_actual_probability",
        "actual_probability_delta_model_minus_market",
        "model_row_log_loss",
        "market_row_log_loss",
        "log_loss_delta_model_minus_market",
        "max_probability_gap",
        "market_favourite",
        "market_favourite_probability",
        "model_market_favourite_probability",
        "market_favourite_probability_gap",
    ]
    return rows.loc[:, [column for column in columns if column in rows.columns]]


def _analysis_columns(extra: list[str] | None = None) -> list[str]:
    columns = [
        "model_name",
        "match_date",
        "home_team",
        "away_team",
        "actual_outcome",
        "model_pick",
        "market_pick",
        "model_actual_probability",
        "market_actual_probability",
        "log_loss_delta_model_minus_market",
        "max_probability_gap",
    ]
    for column in extra or []:
        if column not in columns:
            columns.append(column)
    return columns


def _round_float_columns(df: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    rounded = df.copy()
    for column in rounded.select_dtypes(include=["float"]).columns:
        rounded[column] = rounded[column].round(digits)
    return rounded


def _round_records(records: list[dict[str, Any]], digits: int = 6) -> list[dict[str, Any]]:
    rounded: list[dict[str, Any]] = []
    for record in records:
        copied = {}
        for key, value in record.items():
            if isinstance(value, float) and math.isfinite(value):
                copied[key] = round(value, digits)
            else:
                copied[key] = value
        rounded.append(copied)
    return rounded


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
    return str(value)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)
