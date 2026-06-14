from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evaluation.evaluation_store import (
    build_evaluation_run_record,
    get_latest_evaluation_runs,
    log_evaluation_run,
)
from evaluation.run_evaluation import run_walk_forward_evaluation


@pytest.fixture
def evaluation_result():
    return {
        "run_id": "",
        "generated_at": "2026-06-14T08:00:00+00:00",
        "evaluation_type": "walk_forward",
        "parameters": {
            "start_date": "2025-10-01",
            "alpha": 0.15,
            "decay_days": 60.0,
            "rho": -0.13,
            "fit_rho_each_batch": False,
            "min_training_matches": 10,
            "n_bins": 5,
        },
        "metrics": {
            "n_matches": 2,
            "brier_score": 0.42,
            "log_loss": 0.91,
            "accuracy": 0.5,
            "calibration_bins": [{"bin": "40%-60%", "count": 1}],
            "confidence_buckets": [{"bucket": "medium", "count": 1}],
        },
        "per_match_results": [{"home_team": "Arsenal", "away_team": "Chelsea"}],
    }


@pytest.fixture
def evaluation_df():
    return pd.DataFrame(
        {
            "match_date": pd.to_datetime(
                [
                    "2025-09-07",
                    "2025-09-14",
                    "2025-09-21",
                    "2025-09-28",
                    "2025-10-05",
                    "2025-10-12",
                    "2025-11-01",
                    "2025-11-08",
                    "2025-11-15",
                ]
            ),
            "round_label": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"],
            "home_team": [
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Chelsea",
                "Arsenal",
            ],
            "away_team": [
                "Chelsea",
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Chelsea",
            ],
            "home_xg": [2.5, 1.2, 2.0, 1.8, 2.3, 1.5, 2.1, 1.9, 2.4],
            "away_xg": [1.0, 2.3, 1.1, 2.1, 0.9, 2.2, 1.0, 2.0, 1.1],
            "home_np_xg": [2.5, 1.2, 2.0, 1.8, 2.3, 1.5, 2.1, 1.9, 2.4],
            "away_np_xg": [1.0, 2.3, 1.1, 2.1, 0.9, 2.2, 1.0, 2.0, 1.1],
            "home_goals": [2, 1, 3, 1, 2, 0, 2, 1, 3],
            "away_goals": [0, 2, 0, 2, 0, 1, 1, 2, 0],
        }
    )


def test_build_evaluation_run_record_shape(evaluation_result):
    record = build_evaluation_run_record(
        evaluation_result=evaluation_result,
        run_trigger="manual",
        code_version="abc123",
        notes="baseline",
        data_snapshot={"rows": 9},
    )

    assert record["evaluation_type"] == "walk_forward"
    assert record["start_date"] == "2025-10-01"
    assert record["model_config"]["alpha"] == 0.15
    assert record["aggregate_metrics"]["brier_score"] == 0.42
    assert record["calibration_bins"] == [{"bin": "40%-60%", "count": 1}]
    assert record["confidence_buckets"] == [{"bucket": "medium", "count": 1}]
    assert record["per_match_results"] == [{"home_team": "Arsenal", "away_team": "Chelsea"}]
    assert record["data_snapshot"] == {"rows": 9}
    assert record["run_trigger"] == "manual"
    assert record["code_version"] == "abc123"
    assert record["notes"] == "baseline"


def test_log_evaluation_run_successful_mocked_insert(evaluation_result):
    client = MagicMock()
    client.table.return_value.insert.return_value.execute.return_value = SimpleNamespace(
        data=[{"run_id": "eval-run-123"}]
    )

    run_id = log_evaluation_run(client, evaluation_result, run_trigger="test")

    assert run_id == "eval-run-123"
    client.table.assert_called_once_with("evaluation_runs")
    inserted_record = client.table.return_value.insert.call_args.args[0]
    assert inserted_record["run_trigger"] == "test"
    assert inserted_record["aggregate_metrics"]["n_matches"] == 2


def test_get_latest_evaluation_runs_mocked_retrieval():
    expected = [{"run_id": "eval-run-123", "evaluation_type": "walk_forward"}]
    client = MagicMock()
    client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = (
        SimpleNamespace(data=expected)
    )

    records = get_latest_evaluation_runs(client, n=3)

    assert records == expected
    client.table.assert_called_once_with("evaluation_runs")
    client.table.return_value.select.return_value.order.return_value.limit.assert_called_once_with(3)


def test_run_evaluation_without_persistence(evaluation_df):
    result = run_walk_forward_evaluation(
        evaluation_df,
        start_date="2025-09-21",
        min_training_matches=2,
    )

    assert result["run_id"] == ""
    assert result["metrics"]["n_matches"] > 0


def test_run_evaluation_with_mocked_persistence(evaluation_df):
    client = MagicMock()
    with patch("evaluation.run_evaluation.log_evaluation_run", return_value="eval-run-123") as log_mock:
        result = run_walk_forward_evaluation(
            evaluation_df,
            start_date="2025-09-21",
            min_training_matches=2,
            persist=True,
            client=client,
            run_trigger="test",
            code_version="abc123",
            notes="mocked persistence",
        )

    assert result["run_id"] == "eval-run-123"
    log_mock.assert_called_once()
    _, logged_result = log_mock.call_args.args
    assert logged_result["metrics"]["n_matches"] > 0
    assert log_mock.call_args.kwargs["run_trigger"] == "test"
    assert log_mock.call_args.kwargs["code_version"] == "abc123"
    assert log_mock.call_args.kwargs["notes"] == "mocked persistence"
    assert log_mock.call_args.kwargs["data_snapshot"]["rows"] == len(evaluation_df)
