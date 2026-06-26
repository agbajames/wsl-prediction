from __future__ import annotations

import pandas as pd
import pytest

from evaluation.backtesting import BacktestConfig, build_rolling_folds, run_backtest_for_model
from experiments.registry import available_models, get_model_constructor
from features.team_form import ImprovedTeamFormFeatureBuilder
from models.neural_network import NeuralNetworkChallenger
from scripts.run_neural_network_experiments import run_neural_network_experiments


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


def test_neural_network_predicts_valid_probabilities(matches: pd.DataFrame) -> None:
    model = NeuralNetworkChallenger(
        hidden_units=5,
        max_iter=80,
        min_training_matches=4,
        random_seed=7,
    ).fit(matches.iloc[:8])
    predictions = model.predict(matches.iloc[8:])

    row_sums = predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    assert row_sums.tolist() == pytest.approx([1.0, 1.0])
    assert (predictions[["p_home_win", "p_draw", "p_away_win"]] >= 0).all().all()
    assert (predictions[["p_home_win", "p_draw", "p_away_win"]] <= 1).all().all()
    assert set(predictions["model_name"]) == {"neural_network"}


@pytest.mark.parametrize("hidden_layers", [(), (5,), (6, 3)])
def test_neural_network_architecture_ladder_predicts_probabilities(
    matches: pd.DataFrame,
    hidden_layers: tuple[int, ...],
) -> None:
    model = NeuralNetworkChallenger(
        hidden_layers=hidden_layers,
        max_iter=40,
        min_training_matches=4,
        random_seed=11,
        validation_fraction=0.25,
        early_stopping_patience=5,
    ).fit(matches.iloc[:8])

    predictions = model.predict(matches.iloc[8:])

    assert predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert {"p_home_win", "p_draw", "p_away_win"}.issubset(predictions.columns)
    expected_architecture = "[]" if not hidden_layers else "[" + ",".join(map(str, hidden_layers)) + "]"
    assert predictions["architecture"].iloc[0] == expected_architecture
    assert model.parameter_count > 0


def test_no_hidden_layer_uses_internal_softmax(matches: pd.DataFrame) -> None:
    model = NeuralNetworkChallenger(
        hidden_layers=(),
        max_iter=40,
        min_training_matches=4,
        random_seed=5,
    ).fit(matches.iloc[:8])
    predictions = model.predict(matches.iloc[8:])

    assert set(predictions["fit_mode"]) == {"internal_softmax"}
    assert predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_early_stopping_records_time_aware_validation_split(matches: pd.DataFrame) -> None:
    model = NeuralNetworkChallenger(
        hidden_layers=(5,),
        max_iter=60,
        min_training_matches=4,
        random_seed=5,
        validation_fraction=0.25,
        min_validation_matches=2,
        early_stopping_patience=5,
    ).fit(matches.iloc[:8])
    features = ImprovedTeamFormFeatureBuilder(feature_group="xg").transform_training(matches.iloc[:8])

    diagnostics = model.fit_diagnostics

    assert diagnostics["fit_training_matches"] == 6
    assert diagnostics["validation_matches"] == 2
    assert diagnostics["early_stopping_skipped"] is False
    assert model.scaler_mean is not None
    assert model.scaler_mean.tolist() == pytest.approx(features[:6].mean(axis=0).tolist())


def test_early_stopping_can_run_with_validation(matches: pd.DataFrame) -> None:
    model = NeuralNetworkChallenger(
        hidden_layers=(5,),
        max_iter=60,
        min_training_matches=4,
        random_seed=5,
        validation_fraction=0.25,
        min_validation_matches=2,
        early_stopping_patience=5,
    ).fit(matches.iloc[:8])

    diagnostics = model.fit_diagnostics

    assert diagnostics["fit_training_matches"] == 6
    assert diagnostics["validation_matches"] == 2
    assert diagnostics["early_stopping_skipped"] is False
    assert model.training_history
    assert model.training_history[-1]["validation_loss"] is not None


def test_neural_network_is_deterministic_with_fixed_seed(matches: pd.DataFrame) -> None:
    first = NeuralNetworkChallenger(hidden_units=5, max_iter=80, min_training_matches=4, random_seed=3).fit(
        matches.iloc[:8]
    )
    second = NeuralNetworkChallenger(hidden_units=5, max_iter=80, min_training_matches=4, random_seed=3).fit(
        matches.iloc[:8]
    )

    first_predictions = first.predict(matches.iloc[8:])
    second_predictions = second.predict(matches.iloc[8:])

    assert first_predictions[["p_home_win", "p_draw", "p_away_win"]].to_numpy() == pytest.approx(
        second_predictions[["p_home_win", "p_draw", "p_away_win"]].to_numpy()
    )


def test_scaler_is_fit_on_training_data_only(matches: pd.DataFrame) -> None:
    model = NeuralNetworkChallenger(hidden_units=5, max_iter=10, min_training_matches=4).fit(matches.iloc[:8])
    original_mean = model.scaler_mean
    changed_future = matches.copy()
    changed_future.loc[8, "home_np_xg"] = 99.0
    changed_future.loc[9, "away_np_xg"] = 99.0
    changed = NeuralNetworkChallenger(hidden_units=5, max_iter=10, min_training_matches=4).fit(changed_future.iloc[:8])

    assert original_mean is not None
    assert changed.scaler_mean is not None
    assert original_mean.tolist() == pytest.approx(changed.scaler_mean.tolist())


def test_small_sample_fallback_works(matches: pd.DataFrame) -> None:
    model = NeuralNetworkChallenger(min_training_matches=20).fit(matches.iloc[:4])
    predictions = model.predict(matches.iloc[4:5])

    assert predictions.loc[0, ["p_home_win", "p_draw", "p_away_win"]].sum() == pytest.approx(1.0)
    assert predictions.loc[0, "fit_mode"] == "naive_small_sample_fallback"


def test_registry_integration() -> None:
    assert "neural_network" in available_models()
    model = get_model_constructor("neural_network")()
    assert model.name == "neural_network"
    assert model.architecture == (8,)


def test_output_is_compatible_with_rolling_backtest(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-10-13", test_end="2025-11-03", min_train_matches=4),
    )

    result = run_backtest_for_model(
        lambda: NeuralNetworkChallenger(hidden_units=5, max_iter=80, min_training_matches=4),
        matches,
        folds,
    )

    assert len(result.predictions) == 4
    assert {"fold_id", "p_home_win", "p_draw", "p_away_win"}.issubset(result.predictions.columns)


def test_neural_experiment_runner_rejects_xg_fallback_by_default(matches: pd.DataFrame) -> None:
    results_only = matches.drop(columns=["home_xg", "away_xg", "home_np_xg", "away_np_xg"])
    config = BacktestConfig(test_start="2025-10-13", test_end="2025-11-03", min_train_matches=4)
    folds = build_rolling_folds(results_only, config)

    with pytest.raises(ValueError, match="require xG source columns"):
        run_neural_network_experiments(
            results_only,
            folds=folds,
            backtest_config=config,
            seeds=(42,),
            max_iter=5,
            baseline_models=(),
        )


def test_neural_experiment_runner_allows_explicit_xg_fallback_diagnostic(matches: pd.DataFrame) -> None:
    results_only = matches.drop(columns=["home_xg", "away_xg", "home_np_xg", "away_np_xg"])
    config = BacktestConfig(test_start="2025-10-13", test_end="2025-11-03", min_train_matches=4)
    folds = build_rolling_folds(results_only, config)

    payload = run_neural_network_experiments(
        results_only,
        folds=folds,
        backtest_config=config,
        seeds=(42,),
        max_iter=5,
        baseline_models=(),
        allow_xg_fallback_to_goals=True,
    )

    assert payload["data_schema"]["xg_columns_present"] == []
    assert not payload["summary"].empty


def test_feature_generation_remains_leakage_safe(matches: pd.DataFrame) -> None:
    changed_future = matches.copy()
    changed_future.loc[2, "home_goals"] = 99
    builder = ImprovedTeamFormFeatureBuilder(feature_group="xg")

    original = builder.transform_training(matches.iloc[:3])
    changed = builder.transform_training(changed_future.iloc[:3])

    assert original[2].tolist() == pytest.approx(changed[2].tolist())
