from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.backtesting import BacktestConfig
from experiments.registry import available_models, get_model_constructor
from scripts.run_model_comparison import load_match_csv, main, run_comparison


@pytest.fixture
def matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_date": pd.to_datetime(
                [
                    "2025-09-01",
                    "2025-09-08",
                    "2025-09-15",
                    "2025-09-22",
                    "2025-09-29",
                    "2025-10-06",
                ]
            ),
            "round_label": ["R1", "R2", "R3", "R4", "R5", "R6"],
            "home_team": ["Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal"],
            "home_goals": [2, 1, 3, 0, 2, 1],
            "away_goals": [0, 2, 1, 1, 0, 1],
        }
    )


def test_registry_returns_expected_models() -> None:
    names = available_models()

    assert "champion_dc_xg" in names
    assert "naive_outcome_rate" in names
    assert "elo_baseline" in names
    assert "logistic_regression" in names
    assert get_model_constructor("naive_outcome_rate")().name == "naive_outcome_rate"


def test_unknown_model_name_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_constructor("nope")


def test_comparison_runner_runs_naive_and_elo_on_mocked_data(matches: pd.DataFrame) -> None:
    payload = run_comparison(
        matches,
        model_names=("naive_outcome_rate", "elo_baseline"),
        backtest_config=BacktestConfig(test_start="2025-09-22", test_end="2025-10-06", min_train_matches=2),
        n_bins=4,
        top_n=2,
    )

    assert payload["models"] == ["naive_outcome_rate", "elo_baseline"]
    assert len(payload["folds"]) == 3
    assert "# WSL Model Evaluation Report" in payload["markdown"]
    assert {row["model_name"] for row in payload["prediction_rows"]} == {"naive_outcome_rate", "elo_baseline"}


def test_output_probabilities_are_valid(matches: pd.DataFrame) -> None:
    payload = run_comparison(
        matches,
        model_names=("naive_outcome_rate", "elo_baseline"),
        backtest_config=BacktestConfig(test_start="2025-09-22", test_end="2025-10-06", min_train_matches=2),
    )

    rows = pd.DataFrame(payload["prediction_rows"])
    assert (rows[["p_home_win", "p_draw", "p_away_win"]] >= 0).all().all()
    assert (rows[["p_home_win", "p_draw", "p_away_win"]] <= 1).all().all()
    assert rows[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1).tolist() == pytest.approx([1.0] * len(rows))
    assert set(rows["actual_outcome"]).issubset({"H", "D", "A"})


def test_report_artifacts_are_created(matches: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "matches.csv"
    md_path = tmp_path / "comparison.md"
    json_path = tmp_path / "comparison.json"
    matches.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_model_comparison.py",
            "--csv",
            str(csv_path),
            "--model",
            "naive_outcome_rate",
            "--model",
            "elo_baseline",
            "--test-start",
            "2025-09-22",
            "--test-end",
            "2025-10-06",
            "--min-train-matches",
            "2",
            "--output-md",
            str(md_path),
            "--output-json",
            str(json_path),
        ],
    )

    main()

    assert md_path.exists()
    assert json_path.exists()
    assert "naive_outcome_rate" in md_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["models"] == ["naive_outcome_rate", "elo_baseline"]


def test_load_match_csv_parses_dates(matches: pd.DataFrame, tmp_path: Path) -> None:
    csv_path = tmp_path / "matches.csv"
    matches.to_csv(csv_path, index=False)

    loaded = load_match_csv(csv_path)

    assert pd.api.types.is_datetime64_any_dtype(loaded["match_date"])

