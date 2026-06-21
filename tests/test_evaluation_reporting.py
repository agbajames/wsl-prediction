from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation.calibration import build_calibration_bins, build_confidence_buckets
from evaluation.compare import compare_model_results, summarize_model_results
from evaluation.failure_analysis import worst_misses
from scripts.run_evaluation_report import build_report_summary, render_markdown_report


@pytest.fixture
def prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": ["champion_dc_xg"] * 4,
            "home_team": ["A", "B", "C", "D"],
            "away_team": ["B", "A", "D", "C"],
            "p_home_win": [0.70, 0.20, 0.80, 0.34],
            "p_draw": [0.20, 0.30, 0.10, 0.33],
            "p_away_win": [0.10, 0.50, 0.10, 0.33],
            "actual_outcome": ["H", "A", "A", "D"],
        }
    )


def test_calibration_bins_are_deterministic(prediction_rows: pd.DataFrame) -> None:
    probabilities = prediction_rows[["p_home_win", "p_draw", "p_away_win"]].values.tolist()
    outcomes = prediction_rows["actual_outcome"].tolist()

    first = build_calibration_bins(probabilities, outcomes, n_bins=4, min_bin_size=2)
    second = build_calibration_bins(probabilities, outcomes, n_bins=4, min_bin_size=2)

    assert first == second
    assert sum(row["count"] for row in first) == 4
    assert any(row["is_sparse"] for row in first)


def test_confidence_buckets_work(prediction_rows: pd.DataFrame) -> None:
    probabilities = prediction_rows[["p_home_win", "p_draw", "p_away_win"]].values.tolist()
    outcomes = prediction_rows["actual_outcome"].tolist()

    buckets = build_confidence_buckets(probabilities, outcomes)

    assert [row["bucket"] for row in buckets] == ["low", "medium", "high"]
    assert sum(row["count"] for row in buckets) == 4


def test_comparison_summary_handles_one_model(prediction_rows: pd.DataFrame) -> None:
    summary = summarize_model_results(prediction_rows)

    assert summary["model_name"] == "champion_dc_xg"
    assert summary["n_matches"] == 4
    assert set(summary) == {"model_name", "n_matches", "brier_score", "log_loss", "accuracy"}


def test_comparison_summary_handles_multiple_models(prediction_rows: pd.DataFrame) -> None:
    challenger = prediction_rows.copy()
    challenger["model_name"] = "future_challenger"
    challenger["p_home_win"] = [0.50, 0.30, 0.30, 0.34]
    challenger["p_draw"] = [0.25, 0.30, 0.30, 0.33]
    challenger["p_away_win"] = [0.25, 0.40, 0.40, 0.33]

    comparison = compare_model_results(pd.concat([prediction_rows, challenger], ignore_index=True))

    assert comparison["model_name"].tolist() == ["future_challenger", "champion_dc_xg"]
    assert comparison["rank"].tolist() == [1, 2]


def test_worst_misses_are_ranked_correctly(prediction_rows: pd.DataFrame) -> None:
    misses = worst_misses(prediction_rows, n=2)

    assert misses[0]["home_team"] == "C"
    assert misses[0]["actual_probability"] == pytest.approx(0.10)
    assert misses[0]["row_log_loss"] > misses[1]["row_log_loss"]


def test_report_generation_works_on_minimal_mocked_data(prediction_rows: pd.DataFrame, tmp_path: Path) -> None:
    summary = build_report_summary(prediction_rows, n_bins=4, top_n=2)
    markdown = render_markdown_report(summary)
    output = tmp_path / "report.md"
    output.write_text(markdown, encoding="utf-8")

    assert "# WSL Model Evaluation Report" in markdown
    assert "champion_dc_xg" in markdown
    assert summary["comparison"][0]["n_matches"] == 4
    assert output.read_text(encoding="utf-8").startswith("# WSL Model Evaluation Report")

