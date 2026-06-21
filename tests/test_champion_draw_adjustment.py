from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.draw_adjustment import (
    additive_draw_adjustment,
    build_draw_adjustment_experiment,
    draw_specific_metrics,
    extract_champion_predictions,
    load_model_comparison_payload,
    metric_summary,
    multiplicative_draw_adjustment,
    probabilities_to_unit_interval,
    render_markdown_report,
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
            "actual_outcome": "D",
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
            "p_home_win": 35.0,
            "p_draw": 34.0,
            "p_away_win": 31.0,
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
    assert probabilities[0, 1] == pytest.approx(0.1798, abs=0.0001)


def test_additive_draw_adjustment_preserves_valid_probabilities() -> None:
    adjusted = additive_draw_adjustment([[0.6, 0.2, 0.2]], 0.05)

    assert adjusted[0].sum() == pytest.approx(1.0)
    assert adjusted[0, 1] == pytest.approx(0.25)
    assert np.all(adjusted >= 0.0)


def test_additive_draw_shrinkage_preserves_valid_probabilities() -> None:
    adjusted = additive_draw_adjustment([[0.6, 0.2, 0.2]], -0.05)

    assert adjusted[0].sum() == pytest.approx(1.0)
    assert adjusted[0, 1] == pytest.approx(0.15)
    assert np.all(adjusted >= 0.0)


def test_multiplicative_draw_adjustment_preserves_valid_probabilities() -> None:
    adjusted = multiplicative_draw_adjustment([[0.6, 0.2, 0.2]], 1.15)

    assert adjusted[0].sum() == pytest.approx(1.0)
    assert adjusted[0, 1] > 0.2
    assert np.all(adjusted >= 0.0)


def test_metric_comparison_original_vs_adjusted() -> None:
    probabilities = probabilities_to_unit_interval([[70, 20, 10], [20, 20, 60]])
    original = metric_summary("original", probabilities, ["H", "A"])
    adjusted = metric_summary("adjusted", multiplicative_draw_adjustment(probabilities, 1.05), ["H", "A"])

    assert original["n_matches"] == adjusted["n_matches"] == 2
    assert original["accuracy"] == adjusted["accuracy"] == 1.0


def test_draw_specific_metric_calculation() -> None:
    probabilities = probabilities_to_unit_interval([[30, 40, 30], [20, 20, 60], [35, 34, 31]])

    metrics = draw_specific_metrics(probabilities, ["D", "A", "D"])

    assert metrics["actual_draws"] == 2
    assert metrics["draw_prediction_rate"] == pytest.approx(1 / 3, abs=0.0001)
    assert metrics["draw_recall"] == pytest.approx(0.5)
    assert metrics["draw_log_loss"] is not None


def test_clear_error_when_champion_predictions_absent() -> None:
    rows = [row for row in _rows() if row["model_name"] != "champion_dc_xg"]

    with pytest.raises(ValueError, match="No prediction rows found for champion model"):
        extract_champion_predictions({"prediction_rows": rows})


def test_markdown_report_generation() -> None:
    summary = build_draw_adjustment_experiment({"prediction_rows": _rows()}, top_n=2)

    markdown = render_markdown_report(summary)

    assert "# Champion Draw-Adjustment Experiment" in markdown
    assert "Draw-Adjustment Variants Tested" in markdown
    assert "Recommendation" in markdown


def test_input_parsing(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps({"prediction_rows": _rows()}), encoding="utf-8")

    payload = load_model_comparison_payload(path)

    assert len(payload["prediction_rows"]) == 8
