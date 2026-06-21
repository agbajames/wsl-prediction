from __future__ import annotations

import pandas as pd
import pytest

from evaluation.backtesting import BacktestConfig, build_rolling_folds, run_backtest_for_model
from models.baselines import EloBaseline, NaiveOutcomeRateBaseline


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


def test_naive_model_learns_valid_probabilities(matches: pd.DataFrame) -> None:
    model = NaiveOutcomeRateBaseline().fit(matches.iloc[:4])

    assert model.probabilities[0] > model.probabilities[1]
    assert all(probability > 0 for probability in model.probabilities)
    assert sum(model.probabilities) == pytest.approx(1.0)


def test_naive_predictions_sum_to_one(matches: pd.DataFrame) -> None:
    predictions = NaiveOutcomeRateBaseline().fit(matches.iloc[:4]).predict(matches.iloc[4:])

    row_sums = predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    assert row_sums.tolist() == pytest.approx([1.0, 1.0])
    assert set(predictions["model_name"]) == {"naive_outcome_rate"}


def test_elo_model_updates_ratings_from_results(matches: pd.DataFrame) -> None:
    model = EloBaseline().fit(matches.iloc[:4])

    assert model.ratings
    assert model.ratings["Arsenal"] != pytest.approx(model.initial_rating)
    assert model.ratings["Chelsea"] != pytest.approx(model.initial_rating)


def test_elo_predictions_sum_to_one(matches: pd.DataFrame) -> None:
    predictions = EloBaseline().fit(matches.iloc[:4]).predict(matches.iloc[4:])

    row_sums = predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    assert row_sums.tolist() == pytest.approx([1.0, 1.0])
    assert set(predictions["model_name"]) == {"elo_baseline"}


def test_elo_handles_unseen_teams(matches: pd.DataFrame) -> None:
    model = EloBaseline().fit(matches.iloc[:4])
    fixtures = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2025-10-13"]),
            "round_label": ["R7"],
            "home_team": ["Unseen FC"],
            "away_team": ["Arsenal"],
        }
    )

    predictions = model.predict(fixtures)

    assert predictions.loc[0, "home_rating"] == pytest.approx(model.initial_rating)
    assert predictions.loc[0, ["p_home_win", "p_draw", "p_away_win"]].sum() == pytest.approx(1.0)


def test_baseline_outputs_are_compatible_with_rolling_backtest(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-09-22", test_end="2025-10-06", min_train_matches=2),
    )

    naive_result = run_backtest_for_model(NaiveOutcomeRateBaseline, matches, folds)
    elo_result = run_backtest_for_model(EloBaseline, matches, folds)

    assert len(naive_result.predictions) == 3
    assert len(elo_result.predictions) == 3
    assert {"fold_id", "p_home_win", "p_draw", "p_away_win"}.issubset(naive_result.predictions.columns)
    assert {"fold_id", "p_home_win", "p_draw", "p_away_win"}.issubset(elo_result.predictions.columns)

