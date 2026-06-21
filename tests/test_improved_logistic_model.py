from __future__ import annotations

import pandas as pd
import pytest

from evaluation.backtesting import BacktestConfig, build_rolling_folds, run_backtest_for_model
from experiments.registry import available_models, get_model_constructor
from features.team_form import ImprovedTeamFormFeatureBuilder
from models.logistic import ImprovedLogisticRegressionChallenger


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


def test_feature_builder_uses_only_prior_matches(matches: pd.DataFrame) -> None:
    builder = ImprovedTeamFormFeatureBuilder(feature_group="full", recent_window=3)
    features = builder.transform_training(matches.iloc[:3])
    names = builder.feature_names
    home_points_idx = names.index("home_points_per_match")
    home_recent_xg_idx = names.index("home_recent_xg_diff_per_match")

    assert features[0, home_points_idx] == pytest.approx(builder.default_points_per_match)
    assert features[0, home_recent_xg_idx] == pytest.approx(
        builder.default_xg_for_per_match - builder.default_xg_against_per_match
    )
    assert features[2, home_points_idx] == pytest.approx(builder.default_points_per_match)


def test_feature_builder_does_not_leak_future_result(matches: pd.DataFrame) -> None:
    changed_future = matches.copy()
    changed_future.loc[2, "home_goals"] = 99
    builder = ImprovedTeamFormFeatureBuilder(feature_group="form")
    original = builder.transform_training(matches.iloc[:3])
    changed = builder.transform_training(changed_future.iloc[:3])

    assert original[:2].ravel().tolist() == pytest.approx(changed[:2].ravel().tolist())
    assert original[2].tolist() == pytest.approx(changed[2].tolist())


def test_improved_model_predicts_valid_probabilities(matches: pd.DataFrame) -> None:
    model = ImprovedLogisticRegressionChallenger(min_training_matches=4, solver="internal").fit(matches.iloc[:8])
    predictions = model.predict(matches.iloc[8:])

    row_sums = predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    assert row_sums.tolist() == pytest.approx([1.0, 1.0])
    assert (predictions[["p_home_win", "p_draw", "p_away_win"]] >= 0).all().all()
    assert (predictions[["p_home_win", "p_draw", "p_away_win"]] <= 1).all().all()
    assert set(predictions["model_name"]) == {"improved_logistic_regression"}


def test_feature_group_ablation_changes_exported_features(matches: pd.DataFrame) -> None:
    base = ImprovedLogisticRegressionChallenger(feature_group="base", solver="internal").fit(matches.iloc[:8])
    full = ImprovedLogisticRegressionChallenger(feature_group="full", solver="internal").fit(matches.iloc[:8])

    assert len(full.export_config()["features"]) > len(base.export_config()["features"])
    assert "home_recent_points_per_match" in full.export_config()["features"]


def test_registry_integration() -> None:
    assert "improved_logistic_regression" in available_models()
    assert get_model_constructor("improved_logistic_regression")().name == "improved_logistic_regression"


def test_output_is_compatible_with_rolling_backtest(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-10-13", test_end="2025-11-03", min_train_matches=4),
    )

    result = run_backtest_for_model(
        lambda: ImprovedLogisticRegressionChallenger(min_training_matches=4, solver="internal"),
        matches,
        folds,
    )

    assert len(result.predictions) == 4
    assert {"fold_id", "p_home_win", "p_draw", "p_away_win"}.issubset(result.predictions.columns)
