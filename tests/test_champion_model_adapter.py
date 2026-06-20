from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

import model.wsl_xg_model as champion_impl
from model.wsl_xg_model import ModelConfig
from models.champion_dc_xg import ChampionDCXGModel


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_date": pd.to_datetime(
                [
                    "2025-09-07",
                    "2025-09-14",
                    "2025-09-21",
                    "2025-09-28",
                    "2025-10-05",
                ]
            ),
            "round_label": ["R1", "R2", "R3", "R4", "R5"],
            "home_team": ["Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal"],
            "away_team": ["Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea"],
            "home_xg": [2.5, 1.2, 2.0, 1.8, 2.3],
            "away_xg": [1.0, 2.3, 1.1, 2.1, 0.9],
            "home_np_xg": [2.5, 1.2, 2.0, 1.8, 2.3],
            "away_np_xg": [1.0, 2.3, 1.1, 2.1, 0.9],
            "home_goals": [2, 1, 3, 1, 2],
            "away_goals": [0, 2, 0, 2, 0],
        }
    )


def test_adapter_exposes_expected_metadata() -> None:
    model = ChampionDCXGModel()

    assert model.name == "champion_dc_xg"
    assert model.family == "xg_dixon_coles_poisson"
    assert model.version == "v1"
    assert model.spec["required_input_schema"] == "data.schemas.CHAMPION_REQUIRED_COLUMNS"
    assert model.spec["config"]["alpha"] == ModelConfig().alpha


def test_adapter_can_be_instantiated_with_frozen_defaults() -> None:
    frozen_config = Path("experiments/configs/champion_dc_xg.yaml").read_text()
    model = ChampionDCXGModel.from_config(
        {
            "model_name": "champion_dc_xg",
            "config": {
                "alpha": 0.15,
                "decay_half_life_days": 60.0,
                "rho": -0.13,
                "max_goals": 8,
                "bootstrap_n": 0,
                "rho_behavior": "fixed_default",
            },
        }
    )

    assert "model_name: champion_dc_xg" in frozen_config
    assert "required_input_schema: data.schemas.CHAMPION_REQUIRED_COLUMNS" in frozen_config
    assert model.config == ModelConfig()


def test_adapter_calls_existing_champion_logic(sample_matches: pd.DataFrame) -> None:
    fake_strengths = SimpleNamespace(expected_goals=lambda home, away: (1.4, 1.1))
    fake_prediction = SimpleNamespace(
        to_dict=lambda: {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "match_date": "2025-10-05",
            "round": "R5",
            "p_home_win": 45.0,
            "p_draw": 27.0,
            "p_away_win": 28.0,
        }
    )

    with (
        patch("models.champion_dc_xg.estimate_team_strengths", return_value=fake_strengths) as strengths,
        patch("models.champion_dc_xg.estimate_penalty_rates", return_value=({"Arsenal": 0.05}, {"Chelsea": 0.04})) as penalties,
        patch("models.champion_dc_xg.predict_fixtures", return_value=[fake_prediction]) as predict,
    ):
        model = ChampionDCXGModel().fit(sample_matches.iloc[:4])
        predictions = model.predict(sample_matches.iloc[4:])

    strengths.assert_called_once()
    penalties.assert_called_once()
    predict.assert_called_once()
    assert predictions.loc[0, "model_name"] == "champion_dc_xg"


def test_adapter_uses_fit_rho_when_config_requests_it(sample_matches: pd.DataFrame) -> None:
    with patch("models.champion_dc_xg.fit_rho", return_value=-0.08) as fit_rho:
        model = ChampionDCXGModel(config=ModelConfig(rho=None)).fit(sample_matches.iloc[:4])

    fit_rho.assert_called_once()
    assert model.rho == -0.08


def test_adapter_output_shape_is_evaluation_compatible(sample_matches: pd.DataFrame) -> None:
    model = ChampionDCXGModel().fit(sample_matches.iloc[:4])
    predictions = model.predict(sample_matches.iloc[4:])

    expected_columns = {
        "home_team",
        "away_team",
        "match_date",
        "round",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "model_name",
        "model_family",
        "model_version",
        "rho",
    }
    assert expected_columns.issubset(predictions.columns)
    assert len(predictions) == 1


def test_adapter_split_flow_uses_existing_split_logic(sample_matches: pd.DataFrame) -> None:
    with patch("models.champion_dc_xg.split_played_future", wraps=champion_impl.split_played_future) as split:
        model = ChampionDCXGModel()
        played, future = model.fit_from_dataset(
            sample_matches,
            train_before=pd.Timestamp("2025-10-05"),
            predict_from=pd.Timestamp("2025-10-05"),
            predict_to=pd.Timestamp("2025-10-05"),
        )

    split.assert_called_once()
    assert len(played) == 4
    assert len(future) == 1
