from __future__ import annotations

import pandas as pd
import pytest

from evaluation.backtesting import BacktestConfig, build_rolling_folds, run_backtest_for_model
from experiments.registry import available_models, get_model_constructor
from models.poisson_regression import PoissonRegressionChallenger, win_draw_loss_probabilities


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
                ]
            ),
            "round_label": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],
            "home_team": ["Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal"],
            "home_xg": [2.4, 0.8, 2.2, 0.7, 2.5, 1.1, 2.1, 0.9],
            "away_xg": [0.7, 2.1, 0.8, 1.9, 0.6, 1.0, 0.9, 1.8],
            "home_np_xg": [2.3, 0.7, 2.1, 0.6, 2.4, 1.0, 2.0, 0.8],
            "away_np_xg": [0.6, 2.0, 0.7, 1.8, 0.5, 0.9, 0.8, 1.7],
            "home_goals": [3, 0, 2, 0, 4, 1, 2, 1],
            "away_goals": [0, 2, 0, 2, 0, 1, 1, 2],
        }
    )


def test_model_fits_small_training_dataset(matches: pd.DataFrame) -> None:
    model = PoissonRegressionChallenger(max_iter=2000).fit(matches.iloc[:6])

    assert model.export_config()["resolved_target_source"] == "goals"
    assert model.expected_home_goals("Arsenal", "Chelsea") > 0
    assert model.expected_away_goals("Arsenal", "Chelsea") > 0


def test_predictions_are_valid_probabilities(matches: pd.DataFrame) -> None:
    predictions = PoissonRegressionChallenger(max_iter=2000).fit(matches.iloc[:6]).predict(matches.iloc[6:])

    row_sums = predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    assert row_sums.tolist() == pytest.approx([1.0, 1.0])
    assert (predictions[["p_home_win", "p_draw", "p_away_win"]] >= 0.0).all().all()
    assert (predictions[["p_home_win", "p_draw", "p_away_win"]] <= 1.0).all().all()
    assert set(predictions["model_name"]) == {"poisson_regression"}
    assert {"expected_home_goals", "expected_away_goals", "predicted_outcome"}.issubset(predictions.columns)


def test_stronger_home_attack_or_weaker_away_defence_increases_expected_home_goals(
    matches: pd.DataFrame,
) -> None:
    model = PoissonRegressionChallenger(max_iter=2000).fit(matches.iloc[:6])

    arsenal_home = model.expected_home_goals("Arsenal", "Chelsea")
    chelsea_home = model.expected_home_goals("Chelsea", "Arsenal")

    assert arsenal_home > chelsea_home


def test_unseen_teams_are_handled_safely(matches: pd.DataFrame) -> None:
    model = PoissonRegressionChallenger(max_iter=2000).fit(matches.iloc[:6])
    fixtures = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2025-10-27"]),
            "round_label": ["R9"],
            "home_team": ["Unseen FC"],
            "away_team": ["Another FC"],
        }
    )

    predictions = model.predict(fixtures)

    assert predictions.loc[0, ["p_home_win", "p_draw", "p_away_win"]].sum() == pytest.approx(1.0)
    assert model.min_rate <= predictions.loc[0, "expected_home_goals"] <= model.max_rate
    assert model.min_rate <= predictions.loc[0, "expected_away_goals"] <= model.max_rate


def test_rates_are_positive_and_capped_on_tiny_extreme_sample() -> None:
    tiny = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2025-09-01", "2025-09-08"]),
            "round_label": ["R1", "R2"],
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Arsenal"],
            "home_goals": [12, 0],
            "away_goals": [0, 11],
        }
    )
    model = PoissonRegressionChallenger(ridge_alpha=10.0, max_iter=2000, max_rate=5.0).fit(tiny)

    assert model.min_rate <= model.expected_home_goals("Arsenal", "Chelsea") <= 5.0
    assert model.min_rate <= model.expected_away_goals("Chelsea", "Arsenal") <= 5.0


def test_xg_target_can_be_used_when_available(matches: pd.DataFrame) -> None:
    model = PoissonRegressionChallenger(target_source="xg", max_iter=2000).fit(matches.iloc[:6])

    assert model.export_config()["resolved_target_source"] == "xg"


def test_target_source_falls_back_to_goals(matches: pd.DataFrame) -> None:
    goals_only = matches.drop(columns=["home_xg", "away_xg", "home_np_xg", "away_np_xg"])

    model = PoissonRegressionChallenger(target_source="np_xg", max_iter=2000).fit(goals_only.iloc[:6])

    assert model.export_config()["resolved_target_source"] == "goals"


def test_poisson_probabilities_sum_to_one() -> None:
    p_home, p_draw, p_away = win_draw_loss_probabilities(1.6, 0.9, max_goals=8)

    assert p_home + p_draw + p_away == pytest.approx(1.0)
    assert p_home > p_away


def test_model_integrates_with_registry() -> None:
    assert "poisson_regression" in available_models()
    assert get_model_constructor("poisson_regression")().name == "poisson_regression"


def test_model_outputs_are_compatible_with_rolling_backtest(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-09-22", test_end="2025-10-20", min_train_matches=2),
    )

    result = run_backtest_for_model(PoissonRegressionChallenger, matches, folds)

    assert len(result.predictions) == 5
    assert {"fold_id", "p_home_win", "p_draw", "p_away_win"}.issubset(result.predictions.columns)
