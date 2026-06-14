"""
evaluation/evaluate_logged_predictions.py
-----------------------------------------
Evaluate dashboard-generated prediction_runs without re-running the model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.supabase_client import fetch_match_data, get_supabase_client
from evaluation.evaluation_store import log_evaluation_run
from evaluation.metrics import evaluate_prediction_set, outcome_indices, validate_probabilities

DASHBOARD_TRIGGER_RE = re.compile(r"^dashboard-season-(?P<season>\d{4}-\d{2})-week-(?P<week>\d{2})$")
OUTCOME_LABELS = ("H", "D", "A")


def parse_dashboard_week(run_trigger: str, season: str | None = None) -> int | None:
    match = DASHBOARD_TRIGGER_RE.match(str(run_trigger))
    if not match:
        return None
    if season and match.group("season") != season:
        return None
    return int(match.group("week"))


def week_range(*, week: int | None = None, start_week: int | None = None, end_week: int | None = None) -> tuple[int, int]:
    if week is not None:
        return week, week
    return start_week or 2, end_week or 22


def fetch_prediction_runs(client: Any, limit: int = 1000) -> list[dict[str, Any]]:
    response = (
        client.table("prediction_runs")
        .select(
            "id, created_at, train_before, predict_from, predict_to, model_config, predictions, "
            "team_strengths, rho_fitted, run_trigger"
        )
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def filter_dashboard_replay_runs(
    runs: list[dict[str, Any]],
    *,
    season: str,
    start_week: int,
    end_week: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for run in runs:
        week = parse_dashboard_week(str(run.get("run_trigger", "")), season=season)
        if week is None or week < start_week or week > end_week:
            continue
        copied = dict(run)
        copied["week"] = week
        selected.append(copied)
    return selected


def select_latest_runs_by_week(runs: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if "week" in run and run["week"] is not None:
            grouped[int(run["week"])].append(run)

    latest: dict[int, dict[str, Any]] = {}
    duplicate_warnings: list[dict[str, Any]] = []
    for week, week_runs in grouped.items():
        ordered = sorted(week_runs, key=lambda item: str(item.get("created_at", "")), reverse=True)
        latest[week] = ordered[0]
        if len(ordered) > 1:
            duplicate_warnings.append(
                {
                    "week": week,
                    "selected_run_id": _run_id(ordered[0]),
                    "duplicate_run_ids": [_run_id(run) for run in ordered[1:]],
                    "all_run_ids": [_run_id(run) for run in ordered],
                }
            )
    return latest, sorted(duplicate_warnings, key=lambda item: item["week"])


def _run_id(run: dict[str, Any]) -> str:
    return str(run.get("id") or run.get("run_id") or "")


def _probability_value(record: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in record and record[key] is not None:
            value = float(record[key])
            return value / 100.0 if value > 1.0 else value
    raise ValueError(f"Prediction record missing probability field. Tried: {keys}")


def flatten_prediction_runs(runs_by_week: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for week, run in sorted(runs_by_week.items()):
        predictions = run.get("predictions") or []
        if not isinstance(predictions, list):
            raise ValueError(f"Prediction run {_run_id(run)} has non-list predictions payload.")

        for idx, prediction in enumerate(predictions):
            if not isinstance(prediction, dict):
                raise ValueError(f"Prediction run {_run_id(run)} contains a non-object prediction at index {idx}.")

            rows.append(
                {
                    "week": week,
                    "run_id": _run_id(run),
                    "created_at": run.get("created_at"),
                    "run_trigger": run.get("run_trigger"),
                    "train_before": run.get("train_before"),
                    "predict_from": run.get("predict_from"),
                    "predict_to": run.get("predict_to"),
                    "match_date": str(prediction.get("match_date") or ""),
                    "round_label": str(prediction.get("round") or prediction.get("round_label") or f"R{week}"),
                    "home_team": str(prediction.get("home_team") or ""),
                    "away_team": str(prediction.get("away_team") or ""),
                    "p_home_win": _probability_value(prediction, "p_home_win", "p_home", "home_win"),
                    "p_draw": _probability_value(prediction, "p_draw", "draw"),
                    "p_away_win": _probability_value(prediction, "p_away_win", "p_away", "away_win"),
                }
            )
    return rows


def filter_predictions_to_expected_round(
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only Week N predictions whose round_label is R<N>."""
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    excluded_by_week: dict[int, set[str]] = defaultdict(set)

    for row in prediction_rows:
        expected_round_label = f"R{row['week']}"
        actual_round_label = str(row.get("round_label", ""))
        if actual_round_label != expected_round_label:
            excluded_row = {
                **row,
                "expected_round_label": expected_round_label,
                "actual_round_label": actual_round_label,
            }
            excluded.append(excluded_row)
            excluded_by_week[int(row["week"])].add(actual_round_label)
            continue
        kept.append(row)

    warnings = [
        {
            "week": week,
            "expected_round_label": f"R{week}",
            "excluded_round_labels": sorted(labels),
        }
        for week, labels in sorted(excluded_by_week.items())
    ]
    return kept, excluded, warnings


def prepare_actual_results(df: pd.DataFrame, *, start_week: int, end_week: int) -> list[dict[str, Any]]:
    required = {"match_date", "round_label", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Actual results missing required columns: {sorted(missing)}")

    work = df.copy()
    work["match_date"] = pd.to_datetime(work["match_date"], format="ISO8601", errors="raise").dt.date.astype(str)
    work["week"] = work["round_label"].map(_week_from_round_label)
    work = work[(work["week"] >= start_week) & (work["week"] <= end_week)]
    work = work[work["home_goals"].notna() & work["away_goals"].notna()]

    rows: list[dict[str, Any]] = []
    for record in work.to_dict(orient="records"):
        home_goals = int(record["home_goals"])
        away_goals = int(record["away_goals"])
        rows.append(
            {
                "week": int(record["week"]),
                "match_date": str(record["match_date"]),
                "round_label": str(record["round_label"]),
                "home_team": str(record["home_team"]),
                "away_team": str(record["away_team"]),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "outcome": "H" if home_goals > away_goals else ("D" if home_goals == away_goals else "A"),
            }
        )
    return rows


def _week_from_round_label(round_label: Any) -> int | None:
    match = re.search(r"(\d+)", str(round_label))
    return int(match.group(1)) if match else None


def _match_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("match_date", "")),
        str(record.get("round_label", "")),
        str(record.get("home_team", "")).strip().casefold(),
        str(record.get("away_team", "")).strip().casefold(),
    )


def match_predictions_to_actuals(
    prediction_rows: list[dict[str, Any]],
    actual_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    actual_by_key = {_match_key(row): row for row in actual_rows}
    prediction_keys = {_match_key(row) for row in prediction_rows}
    matched: list[dict[str, Any]] = []
    unmatched_predictions: list[dict[str, Any]] = []

    for prediction in prediction_rows:
        key = _match_key(prediction)
        actual = actual_by_key.get(key)
        if actual is None:
            unmatched_predictions.append(prediction)
            continue
        matched.append({**prediction, **{f"actual_{key}": value for key, value in actual.items()}})

    unmatched_actuals = [actual for actual in actual_rows if _match_key(actual) not in prediction_keys]
    return matched, unmatched_predictions, unmatched_actuals


def evaluate_matched_predictions(matched_rows: list[dict[str, Any]], n_bins: int = 5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not matched_rows:
        raise ValueError("No logged predictions could be matched to completed actual results.")

    probabilities = [
        [row["p_home_win"], row["p_draw"], row["p_away_win"]]
        for row in matched_rows
    ]
    outcomes = [row["actual_outcome"] for row in matched_rows]
    metrics = evaluate_prediction_set(probabilities, outcomes, n_bins=n_bins)
    probs = validate_probabilities(probabilities)
    actual_indices = outcome_indices(outcomes)

    per_match_results: list[dict[str, Any]] = []
    for row, prob_row, actual_idx in zip(matched_rows, probs, actual_indices, strict=True):
        predicted_idx = int(prob_row.argmax())
        actual_probability = float(prob_row[actual_idx])
        confidence = float(prob_row[predicted_idx])
        per_match_results.append(
            {
                "week": row["week"],
                "run_id": row["run_id"],
                "match_date": row["match_date"],
                "round_label": row["round_label"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": row["actual_home_goals"],
                "away_goals": row["actual_away_goals"],
                "actual_outcome": row["actual_outcome"],
                "predicted_outcome": OUTCOME_LABELS[predicted_idx],
                "correct": predicted_idx == int(actual_idx),
                "confidence": round(confidence, 6),
                "actual_outcome_probability": round(actual_probability, 6),
                "p_home_win": round(float(prob_row[0]), 6),
                "p_draw": round(float(prob_row[1]), 6),
                "p_away_win": round(float(prob_row[2]), 6),
            }
        )
    return metrics, per_match_results


def build_logged_evaluation_result(
    *,
    season: str,
    start_week: int,
    end_week: int,
    prediction_runs: list[dict[str, Any]],
    actuals_df: pd.DataFrame,
    n_bins: int = 5,
) -> dict[str, Any]:
    replay_runs = filter_dashboard_replay_runs(
        prediction_runs,
        season=season,
        start_week=start_week,
        end_week=end_week,
    )
    latest_runs, duplicate_warnings = select_latest_runs_by_week(replay_runs)
    prediction_rows = flatten_prediction_runs(latest_runs)
    prediction_rows, excluded_round_mismatches, round_mismatch_warnings = filter_predictions_to_expected_round(
        prediction_rows
    )
    actual_rows = prepare_actual_results(actuals_df, start_week=start_week, end_week=end_week)
    matched_rows, unmatched_predictions, unmatched_actuals = match_predictions_to_actuals(prediction_rows, actual_rows)
    metrics, per_match_results = evaluate_matched_predictions(matched_rows, n_bins=n_bins)

    evaluated_weeks = sorted({row["week"] for row in per_match_results})
    expected_weeks = list(range(start_week, end_week + 1))
    missing_weeks = [week for week in expected_weeks if week not in latest_runs]
    evaluated_fixture_count_by_week = {
        week: sum(1 for row in per_match_results if row["week"] == week)
        for week in expected_weeks
    }
    fixture_count_anomalies = [
        {"week": week, "evaluated_fixture_count": count, "expected_fixture_count": 6}
        for week, count in evaluated_fixture_count_by_week.items()
        if count != 6
    ]
    start_date = min(row["match_date"] for row in per_match_results)
    end_date = max(row["match_date"] for row in per_match_results)
    run_trigger_pattern = f"dashboard-season-{season}-week-XX"

    return {
        "run_id": "",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "logged_prediction_matchweek" if start_week == end_week else "logged_prediction_replay",
        "parameters": {
            "season": season,
            "start_week": start_week,
            "end_week": end_week,
            "start_date": start_date,
            "end_date": end_date,
            "source": "prediction_runs",
            "run_trigger_pattern": run_trigger_pattern,
            "n_bins": n_bins,
        },
        "metrics": metrics,
        "per_match_results": per_match_results,
        "warnings": duplicate_warnings + round_mismatch_warnings,
        "data_snapshot": {
            "prediction_run_count": len(latest_runs),
            "raw_prediction_run_count": len(replay_runs),
            "evaluated_weeks": evaluated_weeks,
            "missing_weeks": missing_weeks,
            "evaluated_fixture_count_by_week": evaluated_fixture_count_by_week,
            "fixture_count_anomalies": fixture_count_anomalies,
            "excluded_round_mismatches": excluded_round_mismatches,
            "unmatched_predictions": unmatched_predictions,
            "unmatched_actuals": unmatched_actuals,
            "duplicate_runs": duplicate_warnings,
            "round_mismatch_warnings": round_mismatch_warnings,
        },
    }


def build_markdown_report(result: dict[str, Any]) -> str:
    params = result["parameters"]
    metrics = result["metrics"]
    per_match = result["per_match_results"]
    snapshot = result["data_snapshot"]
    best = sorted(per_match, key=lambda row: (not row["correct"], -row["confidence"]))[:10]
    worst = sorted(per_match, key=lambda row: row["actual_outcome_probability"])[:10]

    lines = [
        f"# Logged Prediction Replay Evaluation: {params['season']}",
        "",
        "## Summary",
        "",
        f"- Season: {params['season']}",
        f"- Weeks evaluated: {params['start_week']}-{params['end_week']}",
        f"- Prediction run count: {snapshot['prediction_run_count']}",
        f"- Fixture count: {metrics['n_matches']}",
        f"- Brier score: {metrics['brier_score']:.6f}",
        f"- Log loss: {metrics['log_loss']:.6f}",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        "- Week 1 is excluded because it requires historical priors or a previous-season baseline.",
        "- Rescheduled/long-window rounds are filtered by official round_label before scoring.",
        "",
        "## Calibration",
        "",
    ]
    lines.extend(_markdown_table(metrics["calibration_bins"]))
    lines.extend(["", "## Confidence Buckets", ""])
    lines.extend(_markdown_table(metrics["confidence_buckets"]))
    lines.extend(["", "## Best 10 Predictions", ""])
    lines.extend(_markdown_table(_report_rows(best)))
    lines.extend(["", "## Worst 10 Misses", ""])
    lines.extend(_markdown_table(_report_rows(worst)))
    lines.extend(["", "## Duplicate Run Warnings", ""])
    lines.append(json.dumps(result.get("warnings", []), indent=2, sort_keys=True))
    lines.extend(["", "## Evaluated Fixture Count By Week", ""])
    lines.extend(
        _markdown_table(
            [
                {"week": week, "evaluated_fixture_count": count}
                for week, count in snapshot["evaluated_fixture_count_by_week"].items()
            ]
        )
    )
    lines.extend(["", "## Fixture Count Anomalies", ""])
    lines.append(json.dumps(snapshot["fixture_count_anomalies"], indent=2, sort_keys=True))
    lines.extend(["", "## Excluded Round-Label Mismatches", ""])
    lines.append(json.dumps(snapshot["excluded_round_mismatches"], indent=2, sort_keys=True))
    lines.extend(["", "## Unmatched Predictions", ""])
    lines.append(json.dumps(snapshot["unmatched_predictions"], indent=2, sort_keys=True))
    lines.extend(["", "## Unmatched Actuals", ""])
    lines.append(json.dumps(snapshot["unmatched_actuals"], indent=2, sort_keys=True))
    return "\n".join(lines) + "\n"


def _report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "week": row["week"],
            "match_date": row["match_date"],
            "fixture": f"{row['home_team']} vs {row['away_team']}",
            "score": f"{row['home_goals']}-{row['away_goals']}",
            "actual": row["actual_outcome"],
            "predicted": row["predicted_outcome"],
            "correct": row["correct"],
            "confidence": row["confidence"],
            "actual_prob": row["actual_outcome_probability"],
        }
        for row in rows
    ]


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["_None_"]
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def run_logged_prediction_evaluation(
    *,
    season: str,
    week: int | None = None,
    start_week: int | None = None,
    end_week: int | None = None,
    persist: bool = False,
    run_trigger: str = "logged-replay",
    code_version: str | None = None,
    notes: str | None = None,
    output: str | None = None,
    client: Any | None = None,
    prediction_runs: list[dict[str, Any]] | None = None,
    actuals_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    resolved_start_week, resolved_end_week = week_range(week=week, start_week=start_week, end_week=end_week)
    if prediction_runs is None or actuals_df is None or persist:
        client = client or get_supabase_client()
    prediction_runs = prediction_runs if prediction_runs is not None else fetch_prediction_runs(client)
    actuals_df = actuals_df if actuals_df is not None else fetch_match_data(client)

    result = build_logged_evaluation_result(
        season=season,
        start_week=resolved_start_week,
        end_week=resolved_end_week,
        prediction_runs=prediction_runs,
        actuals_df=actuals_df,
    )

    if persist:
        result["run_id"] = log_evaluation_run(
            client,
            result,
            run_trigger=run_trigger,
            code_version=code_version,
            notes=notes,
            data_snapshot=result["data_snapshot"],
        )

    if output:
        with open(output, "w", encoding="utf-8") as file:
            file.write(build_markdown_report(result))

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate logged dashboard prediction_runs.")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--start-week", type=int, default=None)
    parser.add_argument("--end-week", type=int, default=None)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--run-trigger", default="logged-replay")
    parser.add_argument("--code-version", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_logged_prediction_evaluation(
        season=args.season,
        week=args.week,
        start_week=args.start_week,
        end_week=args.end_week,
        persist=args.persist,
        run_trigger=args.run_trigger,
        code_version=args.code_version,
        notes=args.notes,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
