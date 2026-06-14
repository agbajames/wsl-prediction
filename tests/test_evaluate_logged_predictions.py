from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evaluation.evaluate_logged_predictions import (
    build_arg_parser,
    build_logged_evaluation_result,
    build_markdown_report,
    filter_dashboard_replay_runs,
    filter_predictions_to_expected_round,
    flatten_prediction_runs,
    match_predictions_to_actuals,
    parse_dashboard_week,
    prepare_actual_results,
    run_logged_prediction_evaluation,
    select_latest_runs_by_week,
)


def _prediction_run(
    *,
    run_id: str,
    week: int,
    created_at: str,
    predictions: list[dict],
    season: str = "2025-26",
) -> dict:
    return {
        "id": run_id,
        "created_at": created_at,
        "train_before": predictions[0]["match_date"],
        "predict_from": predictions[0]["match_date"],
        "predict_to": predictions[-1]["match_date"],
        "run_trigger": f"dashboard-season-{season}-week-{week:02d}",
        "predictions": predictions,
    }


def _prediction(match_date: str, round_label: str, home: str, away: str, p_home: float, p_draw: float, p_away: float):
    return {
        "match_date": match_date,
        "round": round_label,
        "home_team": home,
        "away_team": away,
        "p_home_win": p_home,
        "p_draw": p_draw,
        "p_away_win": p_away,
    }


@pytest.fixture
def prediction_runs():
    return [
        _prediction_run(
            run_id="run-w2-old",
            week=2,
            created_at="2026-01-01T09:00:00Z",
            predictions=[_prediction("2025-09-12", "R2", "Arsenal", "Chelsea", 40.0, 30.0, 30.0)],
        ),
        _prediction_run(
            run_id="run-w2-new",
            week=2,
            created_at="2026-01-01T10:00:00Z",
            predictions=[
                _prediction("2025-09-12", "R2", "Arsenal", "Chelsea", 70.0, 20.0, 10.0),
                _prediction("2025-09-13", "R2", "Spurs", "Liverpool", 20.0, 20.0, 60.0),
            ],
        ),
        _prediction_run(
            run_id="run-w3",
            week=3,
            created_at="2026-01-02T10:00:00Z",
            predictions=[_prediction("2025-09-19", "R3", "Man City", "Everton", 20.0, 60.0, 20.0)],
        ),
    ]


@pytest.fixture
def actuals_df():
    return pd.DataFrame(
        [
            {
                "match_date": "2025-09-12",
                "round_label": "R2",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_goals": 2,
                "away_goals": 0,
            },
            {
                "match_date": "2025-09-13",
                "round_label": "R2",
                "home_team": "Spurs",
                "away_team": "Liverpool",
                "home_goals": 0,
                "away_goals": 1,
            },
            {
                "match_date": "2025-09-19",
                "round_label": "R3",
                "home_team": "Man City",
                "away_team": "Everton",
                "home_goals": 1,
                "away_goals": 1,
            },
            {
                "match_date": "2025-09-19",
                "round_label": "R3",
                "home_team": "Brighton",
                "away_team": "Leicester",
                "home_goals": 1,
                "away_goals": 0,
            },
        ]
    )


def test_parse_dashboard_week_from_run_trigger():
    assert parse_dashboard_week("dashboard-season-2025-26-week-07", season="2025-26") == 7
    assert parse_dashboard_week("dashboard-season-2024-25-week-07", season="2025-26") is None
    assert parse_dashboard_week("manual") is None


def test_select_latest_prediction_run_when_duplicates_exist(prediction_runs):
    selected = filter_dashboard_replay_runs(prediction_runs, season="2025-26", start_week=2, end_week=3)
    latest, warnings = select_latest_runs_by_week(selected)

    assert latest[2]["id"] == "run-w2-new"
    assert latest[3]["id"] == "run-w3"
    assert warnings == [
        {
            "week": 2,
            "selected_run_id": "run-w2-new",
            "duplicate_run_ids": ["run-w2-old"],
            "all_run_ids": ["run-w2-new", "run-w2-old"],
        }
    ]


def test_flatten_stored_prediction_payloads(prediction_runs):
    selected = filter_dashboard_replay_runs(prediction_runs, season="2025-26", start_week=2, end_week=2)
    latest, _ = select_latest_runs_by_week(selected)

    rows = flatten_prediction_runs(latest)

    assert len(rows) == 2
    assert rows[0]["p_home_win"] == pytest.approx(0.7)
    assert rows[0]["round_label"] == "R2"
    assert rows[0]["run_id"] == "run-w2-new"


def test_week_16_payload_with_r16_and_r20_only_evaluates_r16():
    runs = [
        _prediction_run(
            run_id="run-w16",
            week=16,
            created_at="2026-03-01T10:00:00Z",
            predictions=[
                _prediction("2026-02-13", "R16", "Arsenal", "Chelsea", 70.0, 20.0, 10.0),
                _prediction("2026-05-09", "R20", "Spurs", "Liverpool", 20.0, 20.0, 60.0),
            ],
        )
    ]
    actuals = pd.DataFrame(
        [
            {
                "match_date": "2026-02-13",
                "round_label": "R16",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_goals": 2,
                "away_goals": 0,
            },
            {
                "match_date": "2026-05-09",
                "round_label": "R20",
                "home_team": "Spurs",
                "away_team": "Liverpool",
                "home_goals": 0,
                "away_goals": 1,
            },
        ]
    )

    result = build_logged_evaluation_result(
        season="2025-26",
        start_week=16,
        end_week=16,
        prediction_runs=runs,
        actuals_df=actuals,
    )

    assert result["metrics"]["n_matches"] == 1
    assert result["per_match_results"][0]["round_label"] == "R16"
    assert result["data_snapshot"]["excluded_round_mismatches"][0]["actual_round_label"] == "R20"
    assert result["data_snapshot"]["round_mismatch_warnings"] == [
        {"week": 16, "expected_round_label": "R16", "excluded_round_labels": ["R20"]}
    ]


def test_filter_predictions_to_expected_round_reports_excluded_rows():
    rows = [
        {"week": 16, "round_label": "R16", "home_team": "A", "away_team": "B"},
        {"week": 16, "round_label": "R20", "home_team": "C", "away_team": "D"},
    ]

    kept, excluded, warnings = filter_predictions_to_expected_round(rows)

    assert len(kept) == 1
    assert kept[0]["round_label"] == "R16"
    assert excluded[0]["expected_round_label"] == "R16"
    assert excluded[0]["actual_round_label"] == "R20"
    assert warnings == [{"week": 16, "expected_round_label": "R16", "excluded_round_labels": ["R20"]}]


def test_matching_predictions_to_actual_results(prediction_runs, actuals_df):
    selected = filter_dashboard_replay_runs(prediction_runs, season="2025-26", start_week=2, end_week=3)
    latest, _ = select_latest_runs_by_week(selected)
    predictions = flatten_prediction_runs(latest)
    actuals = prepare_actual_results(actuals_df, start_week=2, end_week=3)

    matched, unmatched_predictions, unmatched_actuals = match_predictions_to_actuals(predictions, actuals)

    assert len(matched) == 3
    assert unmatched_predictions == []
    assert len(unmatched_actuals) == 1
    assert unmatched_actuals[0]["home_team"] == "Brighton"


def test_no_cross_round_actual_matching_is_allowed():
    prediction_rows = [
        {
            "week": 16,
            "run_id": "run-w16",
            "match_date": "2026-05-09",
            "round_label": "R16",
            "home_team": "Spurs",
            "away_team": "Liverpool",
            "p_home_win": 0.2,
            "p_draw": 0.2,
            "p_away_win": 0.6,
        }
    ]
    actual_rows = [
        {
            "week": 20,
            "match_date": "2026-05-09",
            "round_label": "R20",
            "home_team": "Spurs",
            "away_team": "Liverpool",
            "home_goals": 0,
            "away_goals": 1,
            "outcome": "A",
        }
    ]

    matched, unmatched_predictions, unmatched_actuals = match_predictions_to_actuals(prediction_rows, actual_rows)

    assert matched == []
    assert unmatched_predictions == prediction_rows
    assert unmatched_actuals == actual_rows


def test_calculating_metrics_on_logged_predictions(prediction_runs, actuals_df):
    result = build_logged_evaluation_result(
        season="2025-26",
        start_week=2,
        end_week=3,
        prediction_runs=prediction_runs,
        actuals_df=actuals_df,
    )

    assert result["evaluation_type"] == "logged_prediction_replay"
    assert result["metrics"]["n_matches"] == 3
    assert result["metrics"]["accuracy"] == pytest.approx(1.0)
    assert result["data_snapshot"]["prediction_run_count"] == 2
    assert result["data_snapshot"]["evaluated_weeks"] == [2, 3]
    assert result["data_snapshot"]["unmatched_actuals"][0]["home_team"] == "Brighton"
    assert result["warnings"][0]["week"] == 2


def test_unmatched_prediction_is_reported(prediction_runs, actuals_df):
    prediction_runs[1]["predictions"].append(
        _prediction("2025-09-14", "R2", "Unmatched", "Fixture", 50.0, 30.0, 20.0)
    )

    result = build_logged_evaluation_result(
        season="2025-26",
        start_week=2,
        end_week=2,
        prediction_runs=prediction_runs,
        actuals_df=actuals_df,
    )

    assert result["evaluation_type"] == "logged_prediction_matchweek"
    assert result["data_snapshot"]["unmatched_predictions"][0]["home_team"] == "Unmatched"


def test_no_matched_rows_raises_clear_error(prediction_runs, actuals_df):
    with pytest.raises(ValueError, match="No logged predictions"):
        build_logged_evaluation_result(
            season="2025-26",
            start_week=9,
            end_week=9,
            prediction_runs=prediction_runs,
            actuals_df=actuals_df,
        )


def test_full_week_2_to_22_target_metadata(prediction_runs, actuals_df):
    result = build_logged_evaluation_result(
        season="2025-26",
        start_week=2,
        end_week=22,
        prediction_runs=prediction_runs,
        actuals_df=actuals_df,
    )

    assert result["parameters"]["start_week"] == 2
    assert result["parameters"]["end_week"] == 22
    assert 4 in result["data_snapshot"]["missing_weeks"]
    assert 22 in result["data_snapshot"]["missing_weeks"]
    assert result["data_snapshot"]["evaluated_fixture_count_by_week"][2] == 2
    assert result["data_snapshot"]["fixture_count_anomalies"][0] == {
        "week": 2,
        "evaluated_fixture_count": 2,
        "expected_fixture_count": 6,
    }


def test_full_week_2_to_22_evaluates_126_round_strict_fixtures():
    runs = []
    actual_rows = []
    for week in range(2, 23):
        predictions = []
        for fixture_idx in range(6):
            match_date = f"2026-01-{week:02d}"
            home = f"Home {week}-{fixture_idx}"
            away = f"Away {week}-{fixture_idx}"
            predictions.append(_prediction(match_date, f"R{week}", home, away, 70.0, 20.0, 10.0))
            actual_rows.append(
                {
                    "match_date": match_date,
                    "round_label": f"R{week}",
                    "home_team": home,
                    "away_team": away,
                    "home_goals": 2,
                    "away_goals": 0,
                }
            )
        runs.append(
            _prediction_run(
                run_id=f"run-w{week}",
                week=week,
                created_at=f"2026-02-{week:02d}T10:00:00Z",
                predictions=predictions,
            )
        )

    result = build_logged_evaluation_result(
        season="2025-26",
        start_week=2,
        end_week=22,
        prediction_runs=runs,
        actuals_df=pd.DataFrame(actual_rows),
    )

    assert result["metrics"]["n_matches"] == 126
    assert result["data_snapshot"]["fixture_count_anomalies"] == []
    assert all(count == 6 for count in result["data_snapshot"]["evaluated_fixture_count_by_week"].values())


def test_persistence_payload_shape_for_evaluation_runs(prediction_runs, actuals_df):
    client = MagicMock()
    client.table.return_value.insert.return_value.execute.return_value = SimpleNamespace(
        data=[{"run_id": "eval-run-123"}]
    )

    result = run_logged_prediction_evaluation(
        season="2025-26",
        start_week=2,
        end_week=3,
        persist=True,
        run_trigger="logged-replay-2025-26-weeks-02-03",
        client=client,
        prediction_runs=prediction_runs,
        actuals_df=actuals_df,
    )

    assert result["run_id"] == "eval-run-123"
    inserted = client.table.return_value.insert.call_args.args[0]
    assert inserted["evaluation_type"] == "logged_prediction_replay"
    assert inserted["evaluation_params"]["source"] == "prediction_runs"
    assert inserted["data_snapshot"]["prediction_run_count"] == 2
    assert inserted["run_trigger"] == "logged-replay-2025-26-weeks-02-03"


def test_markdown_report_contains_key_sections(prediction_runs, actuals_df):
    result = build_logged_evaluation_result(
        season="2025-26",
        start_week=2,
        end_week=3,
        prediction_runs=prediction_runs,
        actuals_df=actuals_df,
    )

    report = build_markdown_report(result)

    assert "Logged Prediction Replay Evaluation" in report
    assert "Best 10 Predictions" in report
    assert "Worst 10 Misses" in report
    assert "Week 1 is excluded" in report


def test_cli_argument_parsing():
    args = build_arg_parser().parse_args(
        [
            "--season",
            "2025-26",
            "--start-week",
            "2",
            "--end-week",
            "22",
            "--persist",
            "--run-trigger",
            "logged-replay-2025-26-weeks-02-22",
        ]
    )

    assert args.season == "2025-26"
    assert args.start_week == 2
    assert args.end_week == 22
    assert args.persist is True
    assert args.run_trigger == "logged-replay-2025-26-weeks-02-22"


def test_report_output_file_is_written(tmp_path, prediction_runs, actuals_df):
    output = tmp_path / "logged_report.md"
    with patch("evaluation.evaluate_logged_predictions.get_supabase_client", return_value=MagicMock()):
        result = run_logged_prediction_evaluation(
            season="2025-26",
            start_week=2,
            end_week=3,
            output=str(output),
            prediction_runs=prediction_runs,
            actuals_df=actuals_df,
        )

    assert result["metrics"]["n_matches"] == 3
    assert output.read_text(encoding="utf-8").startswith("# Logged Prediction Replay Evaluation")
