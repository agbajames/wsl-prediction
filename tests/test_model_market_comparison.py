from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.model_market_comparison import (
    build_model_market_comparison,
    load_model_prediction_rows,
    match_model_to_market,
    normalize_fixture_date,
    normalize_team_name,
    prepare_model_rows,
    render_model_market_markdown,
    write_model_market_outputs,
)


@pytest.fixture
def market_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2025-10-03", "2025-10-05", "2025-10-12", "2025-10-19"],
            "Home_Team": ["Manchester Utd", "Leicester", "Chelsea", "Arsenal"],
            "Home_Goals": [1, 0, 2, 0],
            "Away_Goals": [1, 2, 1, 1],
            "Away_Team": ["Chelsea", "Everton", "Tottenham", "West Ham"],
            "Odds_1": ["1/1", "3/1", "2/5", "1/2"],
            "Odds_X": ["5/2", "3/1", "4/1", "3/1"],
            "Odds_2": ["3/1", "4/5", "6/1", "6/1"],
            "Imp_Home": [0.5, 0.25, 0.7143, 0.6667],
            "Imp_Draw": [0.2857, 0.25, 0.2, 0.25],
            "Imp_Away": [0.25, 0.5556, 0.1429, 0.1429],
            "Overround": [1.0357, 1.0556, 1.0572, 1.0596],
            "P_Home": [0.4828, 0.2368, 0.6758, 0.629],
            "P_Draw": [0.2759, 0.2368, 0.1892, 0.236],
            "P_Away": [0.2414, 0.5263, 0.1351, 0.135],
            "Note": ["", "", "", ""],
        }
    )


@pytest.fixture
def model_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_date": ["2025-10-03T00:00:00", "2025-10-05", "2025-10-12", "2025-10-26"],
            "home_team": ["Manchester United", "Leicester City", "Chelsea", "Liverpool"],
            "away_team": ["Chelsea", "Everton", "Tottenham Hotspur", "Arsenal"],
            "home_goals": [1, 0, 2, 0],
            "away_goals": [1, 2, 1, 2],
            "actual_outcome": ["D", "A", "H", "A"],
            "model_name": ["champion_dc_xg", "champion_dc_xg", "champion_dc_xg", "champion_dc_xg"],
            "p_home_win": [50.0, 0.40, 0.35, 0.20],
            "p_draw": [25.0, 0.20, 0.30, 0.20],
            "p_away_win": [25.0, 0.40, 0.35, 0.60],
        }
    )


def test_team_alias_normalisation() -> None:
    assert normalize_team_name("Manchester United") == "manchester utd"
    assert normalize_team_name("Tottenham Hotspur") == "tottenham"
    assert normalize_team_name("Leicester City") == "leicester"
    assert normalize_team_name("West Ham United") == "west ham"


def test_date_normalisation() -> None:
    assert normalize_fixture_date("2025-10-03T00:00:00") == "2025-10-03"


def test_prepare_model_rows_scales_percentage_probabilities(model_rows: pd.DataFrame) -> None:
    prepared = prepare_model_rows(model_rows)

    assert prepared.loc[0, "p_home_win"] == pytest.approx(0.50)
    assert prepared[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1).tolist() == pytest.approx([1.0] * 4)


def test_fixture_matching_reports_unmatched_rows(model_rows: pd.DataFrame, market_rows: pd.DataFrame) -> None:
    joined, unmatched_model, unmatched_market = match_model_to_market(model_rows, market_rows)

    assert joined["fixture_key"].nunique() == 3
    assert len(unmatched_model) == 1
    assert len(unmatched_market) == 1
    assert unmatched_model[0]["home_team"] == "Liverpool"
    assert unmatched_market[0]["home_team"] == "Arsenal"


def test_build_model_market_comparison_outputs_sections(
    model_rows: pd.DataFrame,
    market_rows: pd.DataFrame,
) -> None:
    result = build_model_market_comparison(model_rows, market_rows, n_bins=4, top_n=2)

    assert result["data_snapshot"]["matched_fixture_count"] == 3
    assert result["data_snapshot"]["unmatched_model_fixture_count"] == 1
    assert result["data_snapshot"]["unmatched_market_fixture_count"] == 1
    assert {row["model_name"] for row in result["metrics"]} == {"champion_dc_xg", "market_implied_benchmark"}
    assert result["disagreement_analysis"]["summary"][0]["disagreement_count"] >= 1
    assert result["draw_sensitivity"][0]["actual_draw_count"] == 1
    assert result["market_favourite_analysis"]["summary"]["n_fixtures"] == 3
    assert result["row_level_results"][0]["model_p_home_win"] >= 0
    assert "market_p_home_win" in result["row_level_results"][0]


def test_load_model_prediction_rows_from_json(tmp_path: Path, model_rows: pd.DataFrame) -> None:
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"prediction_rows": model_rows.to_dict(orient="records")}), encoding="utf-8")

    loaded = load_model_prediction_rows(path)

    assert len(loaded) == len(model_rows)


def test_report_rendering_uses_safe_language(model_rows: pd.DataFrame, market_rows: pd.DataFrame) -> None:
    result = build_model_market_comparison(model_rows, market_rows, n_bins=4, top_n=2)
    markdown = render_model_market_markdown(result)

    assert "evaluation-only matched-fixture comparison" in markdown
    assert "market-implied" in markdown.lower()
    prohibited = "production " + "betting benchmark"
    assert prohibited not in markdown.lower()


def test_write_outputs(tmp_path: Path, model_rows: pd.DataFrame, market_rows: pd.DataFrame) -> None:
    result = build_model_market_comparison(model_rows, market_rows, n_bins=4, top_n=2)
    md_path = tmp_path / "comparison.md"
    json_path = tmp_path / "comparison.json"
    rows_path = tmp_path / "comparison_rows.csv"

    write_model_market_outputs(result, output_md=md_path, output_json=json_path, output_rows=rows_path)

    assert md_path.read_text(encoding="utf-8").startswith("# WSL Model vs Market-Implied Benchmark")
    assert json.loads(json_path.read_text(encoding="utf-8"))["data_snapshot"]["matched_fixture_count"] == 3
    assert pd.read_csv(rows_path).shape[0] == 3
