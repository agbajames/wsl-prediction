from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.market_benchmark import (
    build_market_benchmark_result,
    load_market_odds_csv,
    normalise_market_rows,
    parse_fractional_or_decimal_odds,
    render_market_benchmark_markdown,
    validate_market_rows,
    write_market_outputs,
)


SAMPLE_PATH = Path("data/samples/market_odds_benchmark_sample.csv")


def test_fractional_odds_are_parsed_to_decimal() -> None:
    assert parse_fractional_or_decimal_odds("39/50") == pytest.approx(1.78)
    assert parse_fractional_or_decimal_odds("4/1") == pytest.approx(5.0)
    assert parse_fractional_or_decimal_odds("2.50") == pytest.approx(2.5)


def test_normalise_market_rows_excludes_non_league_by_default() -> None:
    raw = load_market_odds_csv(SAMPLE_PATH)

    rows = normalise_market_rows(raw)

    assert len(rows) == 4
    assert set(rows["model_name"]) == {"market_implied_benchmark"}
    assert rows.loc[0, "actual_outcome"] == "H"
    assert {"p_home_win", "p_draw", "p_away_win"}.issubset(rows.columns)
    assert {"provided_p_home", "provided_p_draw", "provided_p_away"}.issubset(rows.columns)
    assert {"derived_p_home", "derived_p_draw", "derived_p_away"}.issubset(rows.columns)
    chelsea_leicester = rows.loc[(rows["home_team"] == "Chelsea") & (rows["away_team"] == "Leicester")].iloc[0]
    assert chelsea_leicester["p_away_win"] == pytest.approx(0.0277, abs=1e-4)
    assert chelsea_leicester["provided_p_away"] == pytest.approx(0.2677)
    assert chelsea_leicester["max_abs_provided_vs_derived_diff"] > 0.05


def test_validate_market_rows_reports_quality_warnings() -> None:
    raw = load_market_odds_csv(SAMPLE_PATH)

    validation = validate_market_rows(raw)

    assert validation["status"] == "ok"
    assert validation["row_counts"]["raw_rows"] == 5
    assert validation["row_counts"]["evaluated_rows"] == 4
    warning_types = {warning["type"] for warning in validation["warnings"]}
    assert "non_league_rows_excluded" in warning_types
    assert "underround_rows" in warning_types
    assert "provided_vs_derived_probability_mismatch" in warning_types


def test_validate_market_rows_fails_missing_columns() -> None:
    raw = pd.DataFrame({"Date": ["2025-09-05"]})

    validation = validate_market_rows(raw)

    assert validation["status"] == "error"
    assert validation["errors"][0]["type"] == "missing_columns"


def test_build_market_benchmark_result_returns_metrics_and_rows() -> None:
    raw = load_market_odds_csv(SAMPLE_PATH)

    result = build_market_benchmark_result(raw, n_bins=4, top_n=2)

    assert result["benchmark_type"] == "market_implied_probability_reference"
    assert result["metrics"]["n_matches"] == 4
    assert result["comparison"][0]["model_name"] == "market_implied_benchmark"
    assert len(result["row_level_results"]) == 4
    assert "row_log_loss" in result["row_level_results"][0]
    assert "provided_p_home" in result["row_level_results"][0]
    assert "derived_p_home" in result["row_level_results"][0]


def test_render_market_benchmark_markdown_uses_safe_language() -> None:
    raw = load_market_odds_csv(SAMPLE_PATH)
    result = build_market_benchmark_result(raw, top_n=2)

    markdown = render_market_benchmark_markdown(result)

    assert "# WSL Market-Implied Benchmark" in markdown
    assert "external market probability reference" in markdown
    assert "raw fractional odds" in markdown
    assert "data-quality diagnostics" in markdown
    assert "production decision artifact" in markdown


def test_write_market_outputs_creates_report_artifacts(tmp_path: Path) -> None:
    raw = load_market_odds_csv(SAMPLE_PATH)
    result = build_market_benchmark_result(raw, top_n=2)
    md_path = tmp_path / "market.md"
    json_path = tmp_path / "market.json"
    rows_path = tmp_path / "market_rows.csv"

    write_market_outputs(result, output_md=md_path, output_json=json_path, output_rows=rows_path)

    assert md_path.read_text(encoding="utf-8").startswith("# WSL Market-Implied Benchmark")
    assert json.loads(json_path.read_text(encoding="utf-8"))["model_name"] == "market_implied_benchmark"
    assert pd.read_csv(rows_path).shape[0] == 4
