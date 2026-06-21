from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from evaluation.backtesting import BacktestConfig, build_rolling_folds, run_backtest_for_model
from features.team_form import TeamFormFeatureBuilder
from models.logistic import LogisticRegressionChallenger


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
                    "2025-11-03",
                ]
            ),
            "round_label": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"],
            "home_team": [
                "Arsenal",
                "Chelsea",
                "Brighton",
                "Arsenal",
                "Chelsea",
                "Brighton",
                "Arsenal",
                "Chelsea",
                "Brighton",
                "Arsenal",
            ],
            "away_team": [
                "Chelsea",
                "Brighton",
                "Arsenal",
                "Brighton",
                "Arsenal",
                "Chelsea",
                "Chelsea",
                "Brighton",
                "Arsenal",
                "Brighton",
            ],
            "home_goals": [2, 1, 0, 1, 0, 2, 1, 3, 1, 2],
            "away_goals": [0, 1, 2, 1, 2, 1, 3, 0, 1, 0],
        }
    )


def test_model_can_be_instantiated() -> None:
    model = LogisticRegressionChallenger()

    assert model.name == "logistic_regression"
    assert model.family == "multinomial_logistic_regression"
    assert model.version == "v1"


def test_model_trains_on_mock_historical_matches(matches: pd.DataFrame) -> None:
    model = LogisticRegressionChallenger(min_training_matches=4, solver="internal").fit(matches.iloc[:8])

    assert model.fit_mode == "internal_softmax"


def test_model_predicts_valid_probabilities(matches: pd.DataFrame) -> None:
    model = LogisticRegressionChallenger(min_training_matches=4, solver="internal").fit(matches.iloc[:8])
    predictions = model.predict(matches.iloc[8:])

    assert len(predictions) == 2
    assert set(["p_home_win", "p_draw", "p_away_win"]).issubset(predictions.columns)
    assert (predictions[["p_home_win", "p_draw", "p_away_win"]] >= 0).all().all()
    assert (predictions[["p_home_win", "p_draw", "p_away_win"]] <= 1).all().all()


def test_probabilities_sum_to_one(matches: pd.DataFrame) -> None:
    model = LogisticRegressionChallenger(min_training_matches=4, solver="internal").fit(matches.iloc[:8])
    predictions = model.predict(matches.iloc[8:])

    row_sums = predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    assert row_sums.tolist() == pytest.approx([1.0, 1.0])


def test_missing_optional_features_are_handled(matches: pd.DataFrame) -> None:
    minimal = matches[["match_date", "round_label", "home_team", "away_team", "home_goals", "away_goals"]]

    model = LogisticRegressionChallenger(min_training_matches=4, solver="internal").fit(minimal.iloc[:8])
    predictions = model.predict(minimal.iloc[8:])

    assert len(predictions) == 2
    assert predictions["model_name"].tolist() == ["logistic_regression", "logistic_regression"]


def test_small_sample_fallback_works(matches: pd.DataFrame) -> None:
    model = LogisticRegressionChallenger(min_training_matches=20, solver="internal").fit(matches.iloc[:4])
    predictions = model.predict(matches.iloc[4:5])

    assert model.fit_mode == "naive_small_sample_fallback"
    assert predictions.loc[0, ["p_home_win", "p_draw", "p_away_win"]].sum() == pytest.approx(1.0)


def test_explicit_sklearn_solver_errors_when_dependency_unavailable(matches: pd.DataFrame) -> None:
    if importlib.util.find_spec("sklearn") is not None:
        pytest.skip("scikit-learn is available in this environment")

    with pytest.raises(ImportError, match="scikit-learn is required"):
        LogisticRegressionChallenger(min_training_matches=4, solver="sklearn").fit(matches.iloc[:8])


def test_output_is_compatible_with_rolling_backtest(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-10-13", test_end="2025-11-03", min_train_matches=4),
    )

    result = run_backtest_for_model(
        lambda: LogisticRegressionChallenger(min_training_matches=4, solver="internal"),
        matches,
        folds,
    )

    assert len(result.predictions) == 4
    assert {"fold_id", "p_home_win", "p_draw", "p_away_win"}.issubset(result.predictions.columns)


def test_feature_builder_uses_training_history_before_current_row(matches: pd.DataFrame) -> None:
    builder = TeamFormFeatureBuilder()
    features = builder.transform_training(matches.iloc[:2])

    assert features[0, 0] == pytest.approx(builder.default_points_per_match)
    assert features[1, 1] == pytest.approx(builder.default_points_per_match)

