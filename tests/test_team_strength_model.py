from __future__ import annotations

import pandas as pd
import pytest

from evaluation.backtesting import BacktestConfig, build_rolling_folds, run_backtest_for_model
from experiments.registry import available_models, get_model_constructor
from models.team_strength import RegularisedTeamStrengthModel, win_draw_loss_probabilities


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
            "home_np_xg": [2.4, 0.8, 2.2, 0.7, 2.5, 1.1],
            "away_np_xg": [0.7, 2.1, 0.8, 1.9, 0.6, 1.0],
            "home_xg": [2.4, 0.8, 2.2, 0.7, 2.5, 1.1],
            "away_xg": [0.7, 2.1, 0.8, 1.9, 0.6, 1.0],
            "home_goals": [2, 1, 3, 0, 2, 1],
            "away_goals": [0, 2, 1, 1, 0, 1],
        }
    )


def test_model_fits_small_training_dataset(matches: pd.DataFrame) -> None:
    model = RegularisedTeamStrengthModel().fit(matches.iloc[:4])

    assert model.attack_strengths["Arsenal"] > model.attack_strengths["Chelsea"]
    assert model.export_config()["resolved_strength_source"] == "np_xg"


def test_predictions_are_valid_probabilities(matches: pd.DataFrame) -> None:
    predictions = RegularisedTeamStrengthModel().fit(matches.iloc[:4]).predict(matches.iloc[4:])

    row_sums = predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    assert row_sums.tolist() == pytest.approx([1.0, 1.0])
    assert (predictions[["p_home_win", "p_draw", "p_away_win"]] >= 0.0).all().all()
    assert set(predictions["model_name"]) == {"regularised_team_strength"}
    assert {"expected_home_goals", "expected_away_goals", "predicted_outcome"}.issubset(predictions.columns)


def test_stronger_attack_increases_expected_goals(matches: pd.DataFrame) -> None:
    model = RegularisedTeamStrengthModel().fit(matches.iloc[:4])

    arsenal_home = model.expected_goals("Arsenal", "Chelsea", venue="home")
    chelsea_home = model.expected_goals("Chelsea", "Arsenal", venue="home")

    assert arsenal_home > chelsea_home


def test_shrinkage_pulls_limited_data_teams_towards_league_average(matches: pd.DataFrame) -> None:
    low_shrinkage = RegularisedTeamStrengthModel(shrinkage_matches=0.1).fit(matches.iloc[:2])
    high_shrinkage = RegularisedTeamStrengthModel(shrinkage_matches=20.0).fit(matches.iloc[:2])

    low_gap = abs(low_shrinkage.attack_strengths["Arsenal"] - 1.0)
    high_gap = abs(high_shrinkage.attack_strengths["Arsenal"] - 1.0)

    assert high_gap < low_gap


def test_unseen_teams_use_league_average_strength(matches: pd.DataFrame) -> None:
    model = RegularisedTeamStrengthModel().fit(matches.iloc[:4])
    fixtures = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2025-10-13"]),
            "round_label": ["R7"],
            "home_team": ["Unseen FC"],
            "away_team": ["Another FC"],
        }
    )

    predictions = model.predict(fixtures)

    assert predictions.loc[0, "expected_home_goals"] == pytest.approx(model._home_rate)
    assert predictions.loc[0, "expected_away_goals"] == pytest.approx(model._away_rate)
    assert predictions.loc[0, ["p_home_win", "p_draw", "p_away_win"]].sum() == pytest.approx(1.0)


def test_goals_fallback_when_xg_is_unavailable(matches: pd.DataFrame) -> None:
    goals_only = matches.drop(columns=["home_np_xg", "away_np_xg", "home_xg", "away_xg"])

    model = RegularisedTeamStrengthModel().fit(goals_only.iloc[:4])

    assert model.export_config()["resolved_strength_source"] == "goals"


def test_poisson_probabilities_sum_to_one() -> None:
    p_home, p_draw, p_away = win_draw_loss_probabilities(1.5, 1.0, max_goals=8)

    assert p_home + p_draw + p_away == pytest.approx(1.0)
    assert p_home > p_away


def test_model_integrates_with_registry() -> None:
    assert "regularised_team_strength" in available_models()
    assert get_model_constructor("regularised_team_strength")().name == "regularised_team_strength"


def test_model_outputs_are_compatible_with_rolling_backtest(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-09-22", test_end="2025-10-06", min_train_matches=2),
    )

    result = run_backtest_for_model(RegularisedTeamStrengthModel, matches, folds)

    assert len(result.predictions) == 3
    assert {"fold_id", "p_home_win", "p_draw", "p_away_win"}.issubset(result.predictions.columns)
