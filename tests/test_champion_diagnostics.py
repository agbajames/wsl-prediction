from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.diagnostics import (
    build_champion_diagnostics,
    favourite_draw_breakdown,
    high_confidence_correct,
    high_confidence_misses,
    load_model_comparison_payload,
    render_markdown_report,
    team_error_summary,
)
from evaluation.failure_analysis import scored_prediction_rows


def _rows() -> list[dict[str, object]]:
    champion_rows = [
        {
            "model_name": "champion_dc_xg",
            "fold_id": "fold_001",
            "match_date": "2025-10-01",
            "round": "R5",
            "home_team": "Arsenal",
            "away_team": "Leicester City",
            "p_home_win": 0.82,
            "p_draw": 0.12,
            "p_away_win": 0.06,
            "actual_outcome": "H",
        },
        {
            "model_name": "champion_dc_xg",
            "fold_id": "fold_001",
            "match_date": "2025-10-02",
            "round": "R5",
            "home_team": "Chelsea",
            "away_team": "Everton",
            "p_home_win": 0.75,
            "p_draw": 0.15,
            "p_away_win": 0.10,
            "actual_outcome": "A",
        },
        {
            "model_name": "champion_dc_xg",
            "fold_id": "fold_002",
            "match_date": "2025-10-08",
            "round": "R6",
            "home_team": "Liverpool",
            "away_team": "Manchester City",
            "p_home_win": 0.22,
            "p_draw": 0.18,
            "p_away_win": 0.60,
            "actual_outcome": "A",
        },
        {
            "model_name": "champion_dc_xg",
            "fold_id": "fold_002",
            "match_date": "2025-10-09",
            "round": "R6",
            "home_team": "Brighton",
            "away_team": "Tottenham",
            "p_home_win": 0.30,
            "p_draw": 0.40,
            "p_away_win": 0.30,
            "actual_outcome": "D",
        },
    ]
    challenger_rows = []
    for row in champion_rows:
        challenger = dict(row)
        challenger["model_name"] = "logistic_regression"
        challenger["p_home_win"] = 0.34
        challenger["p_draw"] = 0.33
        challenger["p_away_win"] = 0.33
        challenger_rows.append(challenger)
    return champion_rows + challenger_rows


def test_diagnostic_input_parsing(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps({"prediction_rows": _rows()}), encoding="utf-8")

    payload = load_model_comparison_payload(path)

    assert len(payload["prediction_rows"]) == 8


def test_high_confidence_miss_extraction() -> None:
    scored = scored_prediction_rows([row for row in _rows() if row["model_name"] == "champion_dc_xg"])

    misses = high_confidence_misses(scored, min_confidence=0.6)

    assert len(misses) == 1
    assert misses[0]["home_team"] == "Chelsea"
    assert misses[0]["predicted_outcome"] == "H"


def test_high_confidence_correct_extraction() -> None:
    scored = scored_prediction_rows([row for row in _rows() if row["model_name"] == "champion_dc_xg"])

    correct = high_confidence_correct(scored, min_confidence=0.6)

    assert [row["home_team"] for row in correct] == ["Arsenal", "Liverpool"]


def test_favourite_draw_breakdown() -> None:
    scored = scored_prediction_rows([row for row in _rows() if row["model_name"] == "champion_dc_xg"])

    breakdown = favourite_draw_breakdown(scored)

    labels = {row["favourite_type"] for row in breakdown}
    assert labels == {"away_favourite", "home_favourite", "predicted_draw"}


def test_team_level_summaries() -> None:
    scored = scored_prediction_rows([row for row in _rows() if row["model_name"] == "champion_dc_xg"])

    teams = team_error_summary(scored)

    assert any(row["team"] == "Chelsea" for row in teams)
    assert all("error_rate" in row for row in teams)


def test_markdown_report_generation() -> None:
    summary = build_champion_diagnostics({"prediction_rows": _rows()}, top_n=2)

    markdown = render_markdown_report(summary)

    assert "# Champion Diagnostics Report" in markdown
    assert "High-Confidence Misses" in markdown
    assert "champion_dc_xg" in markdown


def test_clear_failure_when_champion_predictions_missing() -> None:
    rows = [row for row in _rows() if row["model_name"] != "champion_dc_xg"]

    with pytest.raises(ValueError, match="No prediction rows found for champion model"):
        build_champion_diagnostics({"prediction_rows": rows})
