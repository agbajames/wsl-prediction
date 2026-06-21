from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.backtesting import BacktestConfig, build_rolling_folds, run_backtest_for_model
from evaluation.dixon_coles_variants import (
    available_variant_names,
    get_variant_provider,
    get_variant_spec,
    run_dixon_coles_variant_experiment,
)
from models.champion_dc_xg import ChampionDCXGModel
from scripts.run_dixon_coles_variant_experiment import main


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
                    "2025-10-13",
                    "2025-10-20",
                    "2025-10-27",
                ]
            ),
            "round_label": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"],
            "home_team": [
                "Arsenal",
                "Chelsea",
                "Man City",
                "Arsenal",
                "Chelsea",
                "Man City",
                "Arsenal",
                "Chelsea",
                "Man City",
            ],
            "away_team": [
                "Chelsea",
                "Man City",
                "Arsenal",
                "Man City",
                "Arsenal",
                "Chelsea",
                "Chelsea",
                "Man City",
                "Arsenal",
            ],
            "home_xg": [2.5, 1.4, 1.6, 2.1, 1.1, 2.2, 2.4, 1.2, 1.8],
            "away_xg": [1.0, 1.9, 1.5, 1.1, 2.0, 0.9, 0.8, 2.1, 1.7],
            "home_np_xg": [2.3, 1.3, 1.4, 2.0, 1.0, 2.1, 2.2, 1.1, 1.7],
            "away_np_xg": [0.9, 1.8, 1.3, 1.0, 1.8, 0.8, 0.7, 1.9, 1.5],
            "home_goals": [2, 1, 1, 3, 0, 2, 2, 1, 1],
            "away_goals": [0, 2, 1, 1, 2, 0, 0, 2, 2],
        }
    )


def test_variant_registry_exposes_predeclared_grid() -> None:
    names = available_variant_names()

    assert names[0] == "champion_dc_xg"
    assert "dc_fit_rho_each_fold" in names
    assert "dc_score_grid_10" in names
    assert get_variant_spec("dc_alpha_030").config_overrides["alpha"] == 0.30
    with pytest.raises(ValueError, match="Unknown Dixon-Coles variant"):
        get_variant_spec("not_a_variant")


def test_variant_predictions_are_valid_probabilities(matches: pd.DataFrame) -> None:
    model = get_variant_provider("dc_rho_mild_minus_08")().fit(matches.iloc[:6])
    predictions = model.predict(matches.iloc[6:8])

    row_sums = predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    assert row_sums.tolist() == pytest.approx([1.0, 1.0])
    assert (predictions[["p_home_win", "p_draw", "p_away_win"]] >= 0).all().all()
    assert predictions["model_name"].tolist() == ["dc_rho_mild_minus_08", "dc_rho_mild_minus_08"]


def test_reference_variant_matches_unchanged_champion_behaviour(matches: pd.DataFrame) -> None:
    champion_predictions = ChampionDCXGModel().fit(matches.iloc[:6]).predict(matches.iloc[6:8])
    reference_predictions = get_variant_provider("champion_dc_xg")().fit(matches.iloc[:6]).predict(matches.iloc[6:8])

    columns = ["xG_home", "xG_away", "np_xG_home", "np_xG_away", "rho"]
    pd.testing.assert_frame_equal(
        champion_predictions[columns].reset_index(drop=True),
        reference_predictions[columns].reset_index(drop=True),
    )
    assert (champion_predictions[["p_home_win", "p_draw", "p_away_win"]] / 100).to_numpy() == pytest.approx(
        reference_predictions[["p_home_win", "p_draw", "p_away_win"]].to_numpy()
    )
    assert reference_predictions["model_name"].tolist() == ["champion_dc_xg", "champion_dc_xg"]


def test_variant_outputs_are_compatible_with_rolling_backtest(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-09-22", test_end="2025-10-27", min_train_matches=3),
    )

    result = run_backtest_for_model(get_variant_provider("dc_score_grid_10"), matches, folds)

    assert len(result.predictions) == 6
    assert {"fold_id", "p_home_win", "p_draw", "p_away_win"}.issubset(result.predictions.columns)
    assert result.model_name == "dc_score_grid_10"


def test_variant_experiment_runs_shared_comparison(matches: pd.DataFrame) -> None:
    payload = run_dixon_coles_variant_experiment(
        matches,
        backtest_config=BacktestConfig(test_start="2025-09-22", test_end="2025-10-27", min_train_matches=3),
        variant_names=("champion_dc_xg", "dc_rho_mild_minus_08"),
        n_bins=4,
        top_n=2,
    )

    assert payload["variants"][0]["model_name"] == "champion_dc_xg"
    assert {row["model_name"] for row in payload["prediction_rows"]} == {
        "champion_dc_xg",
        "dc_rho_mild_minus_08",
    }
    assert "# Dixon-Coles Configuration Variant Experiment" in payload["markdown"]


def test_cli_writes_report_artifacts(matches: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "matches.csv"
    md_path = tmp_path / "dc_variants.md"
    json_path = tmp_path / "dc_variants.json"
    matches.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_dixon_coles_variant_experiment.py",
            "--csv",
            str(csv_path),
            "--variant",
            "champion_dc_xg",
            "--variant",
            "dc_rho_mild_minus_08",
            "--test-start",
            "2025-09-22",
            "--test-end",
            "2025-10-27",
            "--min-train-matches",
            "3",
            "--output-md",
            str(md_path),
            "--output-json",
            str(json_path),
        ],
    )

    main()

    assert md_path.exists()
    assert json_path.exists()
    assert "dc_rho_mild_minus_08" in md_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["variants"][0]["model_name"] == "champion_dc_xg"
