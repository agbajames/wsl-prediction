from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.calibrators import (
    build_calibration_experiment,
    extract_champion_predictions,
    fit_temperature_shrinkage_calibrator,
    load_model_comparison_payload,
    metric_summary,
    probabilities_to_unit_interval,
    render_markdown_report,
    temperature_scale_probabilities,
)


def _rows() -> list[dict[str, object]]:
    champion_rows = [
        {
            "model_name": "champion_dc_xg",
            "fold_id": "fold_001",
            "match_date": "2025-10-01",
            "home_team": "Arsenal",
            "away_team": "Everton",
            "p_home_win": 70.0,
            "p_draw": 20.0,
            "p_away_win": 10.0,
            "actual_outcome": "H",
        },
        {
            "model_name": "champion_dc_xg",
            "fold_id": "fold_001",
            "match_date": "2025-10-02",
            "home_team": "Chelsea",
            "away_team": "Liverpool",
            "p_home_win": 60.0,
            "p_draw": 25.0,
            "p_away_win": 15.0,
            "actual_outcome": "H",
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
        {
            "model_name": "champion_dc_xg",
            "fold_id": "fold_003",
            "match_date": "2025-10-16",
            "home_team": "Aston Villa",
            "away_team": "West Ham United",
            "p_home_win": 0.40,
            "p_draw": 0.30,
            "p_away_win": 0.30,
            "actual_outcome": "A",
        },
        {
            "model_name": "champion_dc_xg",
            "fold_id": "fold_003",
            "match_date": "2025-10-17",
            "home_team": "Manchester United",
            "away_team": "London City Lionesses",
            "p_home_win": 0.45,
            "p_draw": 0.30,
            "p_away_win": 0.25,
            "actual_outcome": "H",
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

    assert len(champion) == 6
    assert champion["model_name"].unique().tolist() == ["champion_dc_xg"]
    assert champion[["p_home_win", "p_draw", "p_away_win"]].max().max() <= 1.0


def test_percentage_probabilities_convert_to_unit_interval() -> None:
    probabilities = probabilities_to_unit_interval([[56.1, 18.0, 26.0]])

    assert probabilities[0].sum() == pytest.approx(1.0)
    assert probabilities[0, 0] == pytest.approx(0.5604, abs=0.0001)


def test_temperature_scaling_keeps_probabilities_valid() -> None:
    scaled = temperature_scale_probabilities([[0.7, 0.2, 0.1]], temperature=1.5)

    assert scaled.shape == (1, 3)
    assert scaled.sum(axis=1)[0] == pytest.approx(1.0)
    assert np.all(scaled >= 0.0)


def test_fit_temperature_shrinkage_transform_sums_to_one() -> None:
    probabilities = probabilities_to_unit_interval([[70, 20, 10], [20, 20, 60], [45, 30, 25]])
    calibrator = fit_temperature_shrinkage_calibrator(probabilities, ["H", "A", "D"])

    calibrated = calibrator.transform(probabilities)

    assert calibrated.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert calibrator.temperature > 0


def test_metric_comparison_original_vs_calibrated() -> None:
    probabilities = probabilities_to_unit_interval([[70, 20, 10], [20, 20, 60]])
    original = metric_summary("original", probabilities, ["H", "A"])
    calibrated = metric_summary("calibrated", probabilities, ["H", "A"])

    assert original["brier_score"] == pytest.approx(calibrated["brier_score"])
    assert original["accuracy"] == 1.0


def test_clear_error_when_champion_predictions_absent() -> None:
    rows = [row for row in _rows() if row["model_name"] != "champion_dc_xg"]

    with pytest.raises(ValueError, match="No prediction rows found for champion model"):
        extract_champion_predictions({"prediction_rows": rows})


def test_markdown_generation() -> None:
    summary = build_calibration_experiment({"prediction_rows": _rows()}, calibration_fraction=0.67, top_n=2)

    markdown = render_markdown_report(summary)

    assert "# Champion Calibration Experiment" in markdown
    assert "Original Champion Vs Calibrated Champion" in markdown
    assert "Recommendation" in markdown


def test_diagnostic_input_parsing(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps({"prediction_rows": _rows()}), encoding="utf-8")

    payload = load_model_comparison_payload(path)

    assert len(payload["prediction_rows"]) == 12
