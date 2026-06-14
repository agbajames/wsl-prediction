from __future__ import annotations

import math

import pandas as pd
import pytest

from evaluation.metrics import (
    brier_score_3way,
    calibration_bins,
    confidence_bucket_summary,
    multiclass_log_loss,
    outcome_accuracy,
    validate_probabilities,
)
from evaluation.run_evaluation import run_walk_forward_evaluation


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


def test_brier_score_3way_correctness():
    probabilities = [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]]
    outcomes = ["H", "A"]

    expected = (((0.7 - 1) ** 2 + 0.2**2 + 0.1**2) + (0.2**2 + 0.5**2 + (0.3 - 1) ** 2)) / 2

    assert brier_score_3way(probabilities, outcomes) == pytest.approx(expected)


def test_multiclass_log_loss_correctness():
    probabilities = [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]]
    outcomes = ["H", "A"]

    expected = -(math.log(0.7) + math.log(0.3)) / 2

    assert multiclass_log_loss(probabilities, outcomes) == pytest.approx(expected)


def test_outcome_accuracy():
    probabilities = [[0.7, 0.2, 0.1], [0.6, 0.2, 0.2], [0.2, 0.5, 0.3]]
    outcomes = ["H", "A", "D"]

    assert outcome_accuracy(probabilities, outcomes) == pytest.approx(2 / 3)


def test_calibration_bin_shape():
    probabilities = [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.34, 0.33, 0.33]]
    outcomes = ["H", "D", "A"]

    bins = calibration_bins(probabilities, outcomes, n_bins=5)

    assert len(bins) == 5
    assert set(bins[0]) == {"bin", "lower", "upper", "count", "mean_confidence", "observed_accuracy"}
    assert sum(bucket["count"] for bucket in bins) == 3


def test_confidence_bucket_summary_shape():
    probabilities = [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.34, 0.33, 0.33]]
    outcomes = ["H", "D", "A"]

    buckets = confidence_bucket_summary(probabilities, outcomes)

    assert [bucket["bucket"] for bucket in buckets] == ["low", "medium", "high"]
    assert sum(bucket["count"] for bucket in buckets) == 3


def test_invalid_probability_handling():
    with pytest.raises(ValueError, match="non-negative"):
        validate_probabilities([[0.8, -0.1, 0.3]])

    with pytest.raises(ValueError, match="finite"):
        validate_probabilities([[0.8, float("nan"), 0.2]])

    with pytest.raises(ValueError, match="three columns"):
        validate_probabilities([[0.8, 0.2]])


def test_evaluation_runner_smoke_with_local_dataframe(evaluation_df):
    result = run_walk_forward_evaluation(
        evaluation_df,
        start_date="2025-09-21",
        min_training_matches=2,
        n_bins=4,
    )

    assert result["evaluation_type"] == "walk_forward"
    assert result["parameters"]["start_date"] == "2025-09-21"
    assert result["metrics"]["n_matches"] > 0
    assert len(result["metrics"]["calibration_bins"]) == 4
    assert result["per_match_results"]
