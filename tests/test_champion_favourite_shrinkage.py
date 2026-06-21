from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.favourite_shrinkage import (
    build_favourite_shrinkage_experiment,
    extract_champion_predictions,
    high_confidence_favourite_metrics,
    home_away_favourite_summary,
    load_model_comparison_payload,
    metric_summary,
    probabilities_to_unit_interval,
    render_markdown_report,
    soft_cap_favourite_shrinkage,
    threshold_favourite_shrinkage,
)


def _rows() -> list[dict[str, object]]:
    champion_rows = [
        {
            "model_name": "champion_dc_xg",
            "fold_id": "fold_001",
            "match_date": "2025-10-01",
            "home_team": "Arsenal",
            "away_team": "Everton",
            "p_home_win": 80.0,
            "p_draw": 12.0,
            "p_away_win": 8.0,
            "actual_outcome": "H",
        },
        {
            "model_name": "champion_dc_xg",
            "fold_id": "fold_001",
            "match_date": "2025-10-02",
            "home_team": "Chelsea",
            "away_team": "Liverpool",
            "p_home_win": 72.0,
            "p_draw": 18.0,
            "p_away_win": 10.0,
            "actual_outcome": "A",
        },
        {
            "model_name": "champion_dc_xg",
            "fold_id": "fold_002",
            "match_date": "2025-10-09",
            "home_team": "Brighton",
            "away_team": "Manchester City",
            "p_home_win": 10.0,
            "p_draw": 20.0,
            "p_away_win": 70.0,
            "actual_outcome": "A",
        },
        {
            "model_name": "champion_dc_xg",
            "fold_id": "fold_002",
            "match_date": "2025-10-10",
            "home_team": "Tottenham",
            "away_team": "Leicester City",
            "p_home_win": 55.0,
            "p_draw": 25.0,
            "p_away_win": 20.0,
            "actual_outcome": "D",
        },
    ]
    other_rows = []
    for row in champion_rows:
        other = dict(row)
        other["model_name"] = "logistic_regression"
        other_rows.append(other)
    return champion_rows + other_rows


def test_extracting_champion_predictions_normalizes_probabilities() -> None:
    champion = extract_champion_predictions({"prediction_rows": _rows()})

    assert len(champion) == 4
    assert champion["model_name"].unique().tolist() == ["champion_dc_xg"]
    assert champion[["p_home_win", "p_draw", "p_away_win"]].max().max() <= 1.0


def test_percentage_probabilities_convert_to_unit_interval() -> None:
    probabilities = probabilities_to_unit_interval([[56.1, 18.0, 26.0]])

    assert probabilities[0].sum() == pytest.approx(1.0)
    assert probabilities[0, 0] == pytest.approx(0.5604, abs=0.0001)


def test_threshold_shrinkage_preserves_valid_probabilities() -> None:
    adjusted = threshold_favourite_shrinkage([[0.8, 0.12, 0.08]], threshold=0.65, strength=0.1)

    assert adjusted[0].sum() == pytest.approx(1.0)
    assert adjusted[0, 0] == pytest.approx(0.7)
    assert np.all(adjusted >= 0.0)


def test_soft_cap_shrinkage_preserves_valid_probabilities() -> None:
    adjusted = soft_cap_favourite_shrinkage([[0.8, 0.12, 0.08]], cap=0.7, strength=0.5)

    assert adjusted[0].sum() == pytest.approx(1.0)
    assert adjusted[0, 0] == pytest.approx(0.75)
    assert np.all(adjusted >= 0.0)


def test_no_change_below_threshold_or_cap() -> None:
    probabilities = [[0.55, 0.25, 0.20]]

    thresholded = threshold_favourite_shrinkage(probabilities, threshold=0.65, strength=0.1)
    capped = soft_cap_favourite_shrinkage(probabilities, cap=0.70, strength=0.5)

    assert np.allclose(thresholded, probabilities)
    assert np.allclose(capped, probabilities)


def test_metric_comparison_original_vs_adjusted() -> None:
    probabilities = probabilities_to_unit_interval([[80, 12, 8], [20, 20, 60]])
    original = metric_summary("original", probabilities, ["H", "A"])
    adjusted = metric_summary(
        "adjusted",
        threshold_favourite_shrinkage(probabilities, threshold=0.65, strength=0.05),
        ["H", "A"],
    )

    assert original["n_matches"] == adjusted["n_matches"] == 2
    assert original["accuracy"] == adjusted["accuracy"] == 1.0


def test_high_confidence_favourite_metric_calculation() -> None:
    probabilities = probabilities_to_unit_interval([[80, 12, 8], [72, 18, 10], [10, 20, 70]])

    metrics = high_confidence_favourite_metrics(probabilities, ["H", "A", "A"], threshold=0.65)

    assert metrics["high_confidence_favourites"] == 3
    assert metrics["high_confidence_miss_count"] == 1
    assert metrics["high_confidence_correct_count"] == 2
    assert metrics["high_confidence_miss_log_loss"] is not None


def test_home_favourite_vs_away_favourite_summary() -> None:
    probabilities = probabilities_to_unit_interval([[80, 12, 8], [10, 20, 70], [30, 40, 30]])

    summary = home_away_favourite_summary(probabilities, ["H", "A", "D"])

    assert {row["favourite_type"] for row in summary} == {"home_favourite", "draw_favourite", "away_favourite"}
    assert all("mean_confidence" in row for row in summary)


def test_clear_error_when_champion_predictions_absent() -> None:
    rows = [row for row in _rows() if row["model_name"] != "champion_dc_xg"]

    with pytest.raises(ValueError, match="No prediction rows found for champion model"):
        extract_champion_predictions({"prediction_rows": rows})


def test_markdown_report_generation() -> None:
    summary = build_favourite_shrinkage_experiment({"prediction_rows": _rows()}, top_n=2)

    markdown = render_markdown_report(summary)

    assert "# Champion Favourite-Shrinkage Experiment" in markdown
    assert "Favourite-Shrinkage Variants Tested" in markdown
    assert "Recommendation" in markdown


def test_input_parsing(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps({"prediction_rows": _rows()}), encoding="utf-8")

    payload = load_model_comparison_payload(path)

    assert len(payload["prediction_rows"]) == 8
