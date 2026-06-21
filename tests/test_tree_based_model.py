from __future__ import annotations

import pandas as pd
import pytest

from evaluation.backtesting import BacktestConfig, build_rolling_folds, run_backtest_for_model
from experiments.registry import available_models, get_model_constructor
from features.team_form import ImprovedTeamFormFeatureBuilder
from models.tree_based import RandomForestChallenger


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
            "home_xg": [2.4, 1.1, 0.7, 1.5, 0.8, 1.8, 1.2, 2.3, 1.0, 2.1],
            "away_xg": [0.7, 1.0, 2.0, 1.2, 2.2, 1.0, 2.1, 0.6, 1.2, 0.8],
            "home_np_xg": [2.2, 1.0, 0.6, 1.4, 0.7, 1.7, 1.1, 2.1, 0.9, 2.0],
            "away_np_xg": [0.6, 0.9, 1.9, 1.1, 2.0, 0.9, 2.0, 0.5, 1.1, 0.7],
            "home_goals": [2, 1, 0, 1, 0, 2, 1, 3, 1, 2],
            "away_goals": [0, 1, 2, 1, 2, 1, 3, 0, 1, 0],
        }
    )


def test_random_forest_predicts_valid_probabilities(matches: pd.DataFrame) -> None:
    model = RandomForestChallenger(
        n_estimators=12,
        max_depth=2,
        min_samples_leaf=2,
        min_training_matches=4,
    ).fit(matches.iloc[:8])
    predictions = model.predict(matches.iloc[8:])

    row_sums = predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    assert row_sums.tolist() == pytest.approx([1.0, 1.0])
    assert (predictions[["p_home_win", "p_draw", "p_away_win"]] >= 0).all().all()
    assert (predictions[["p_home_win", "p_draw", "p_away_win"]] <= 1).all().all()
    assert set(predictions["model_name"]) == {"random_forest"}


def test_random_forest_is_deterministic_with_fixed_seed(matches: pd.DataFrame) -> None:
    first = RandomForestChallenger(
        n_estimators=12,
        max_depth=2,
        min_samples_leaf=2,
        min_training_matches=4,
        random_seed=7,
    ).fit(matches.iloc[:8])
    second = RandomForestChallenger(
        n_estimators=12,
        max_depth=2,
        min_samples_leaf=2,
        min_training_matches=4,
        random_seed=7,
    ).fit(matches.iloc[:8])

    first_predictions = first.predict(matches.iloc[8:])
    second_predictions = second.predict(matches.iloc[8:])

    assert first_predictions[["p_home_win", "p_draw", "p_away_win"]].to_numpy() == pytest.approx(
        second_predictions[["p_home_win", "p_draw", "p_away_win"]].to_numpy()
    )


def test_feature_importance_is_reported(matches: pd.DataFrame) -> None:
    model = RandomForestChallenger(
        n_estimators=12,
        max_depth=2,
        min_samples_leaf=2,
        min_training_matches=4,
    ).fit(matches.iloc[:8])

    assert set(model.feature_importances).issubset(set(model.export_config()["features"]))
    assert sum(model.feature_importances.values()) == pytest.approx(1.0)


def test_small_sample_fallback_works(matches: pd.DataFrame) -> None:
    model = RandomForestChallenger(min_training_matches=20).fit(matches.iloc[:4])
    predictions = model.predict(matches.iloc[4:5])

    assert predictions.loc[0, ["p_home_win", "p_draw", "p_away_win"]].sum() == pytest.approx(1.0)
    assert predictions.loc[0, "fit_mode"] == "naive_small_sample_fallback"


def test_registry_integration() -> None:
    assert "random_forest" in available_models()
    assert get_model_constructor("random_forest")().name == "random_forest"


def test_output_is_compatible_with_rolling_backtest(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-10-13", test_end="2025-11-03", min_train_matches=4),
    )

    result = run_backtest_for_model(
        lambda: RandomForestChallenger(n_estimators=12, max_depth=2, min_samples_leaf=2, min_training_matches=4),
        matches,
        folds,
    )

    assert len(result.predictions) == 4
    assert {"fold_id", "p_home_win", "p_draw", "p_away_win"}.issubset(result.predictions.columns)


def test_feature_generation_remains_leakage_safe(matches: pd.DataFrame) -> None:
    changed_future = matches.copy()
    changed_future.loc[2, "home_goals"] = 99
    builder = ImprovedTeamFormFeatureBuilder(feature_group="xg")

    original = builder.transform_training(matches.iloc[:3])
    changed = builder.transform_training(changed_future.iloc[:3])

    assert original[2].tolist() == pytest.approx(changed[2].tolist())
