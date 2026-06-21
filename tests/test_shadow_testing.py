from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from evaluation.shadow import (
    DEFAULT_SHADOW_MODELS,
    evaluate_shadow_predictions,
    generate_shadow_predictions,
    load_fixture_file,
    load_shadow_predictions,
    validate_shadow_model_names,
    validate_shadow_prediction_frame,
    write_shadow_predictions,
)


@dataclass
class ConstantProbabilityModel:
    name_value: str
    probabilities: tuple[float, float, float]

    @property
    def name(self) -> str:
        return self.name_value

    @property
    def family(self) -> str:
        return "test_constant_probability"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def spec(self) -> dict[str, Any]:
        return self.export_config()

    def fit(self, played: pd.DataFrame) -> ConstantProbabilityModel:
        assert played["match_date"].max() < pd.Timestamp("2026-09-06")
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "home_team": fixtures["home_team"].tolist(),
                "away_team": fixtures["away_team"].tolist(),
                "match_date": fixtures["match_date"].astype(str).tolist(),
                "p_home_win": self.probabilities[0],
                "p_draw": self.probabilities[1],
                "p_away_win": self.probabilities[2],
                "model_name": self.name,
                "model_family": self.family,
                "model_version": self.version,
            }
        )

    def export_config(self) -> dict[str, Any]:
        return {"model_name": self.name, "model_family": self.family, "model_version": self.version}


@pytest.fixture
def historical_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2026-08-01", "2026-08-08", "2026-08-15"]),
            "home_team": ["Arsenal", "Chelsea", "Manchester City"],
            "away_team": ["Chelsea", "Manchester City", "Arsenal"],
            "home_xg": [2.1, 1.3, 1.8],
            "away_xg": [0.9, 1.7, 1.2],
            "home_np_xg": [2.0, 1.2, 1.7],
            "away_np_xg": [0.8, 1.6, 1.1],
            "home_goals": [2, 1, 2],
            "away_goals": [0, 2, 1],
        }
    )


@pytest.fixture
def upcoming_fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fixture_id": ["fixture-1", "fixture-2"],
            "fixture_date": ["2026-09-06", "2026-09-06"],
            "round_label": ["R1", "R1"],
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Manchester City"],
        }
    )


def test_default_shadow_tracking_set_keeps_champion_reference() -> None:
    assert DEFAULT_SHADOW_MODELS[0] == "champion_dc_xg"
    assert "blend_dc_fit_txg_50_50" in DEFAULT_SHADOW_MODELS


def test_shadow_model_name_validation_rejects_unknown_candidate() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        validate_shadow_model_names(("champion_dc_xg", "not_a_model"))


def test_shadow_prediction_artifact_schema_and_timestamp(
    historical_matches: pd.DataFrame,
    upcoming_fixtures: pd.DataFrame,
) -> None:
    predictions = generate_shadow_predictions(
        historical_matches,
        upcoming_fixtures,
        model_names=("constant_home",),
        model_providers={"constant_home": lambda: ConstantProbabilityModel("constant_home", (0.6, 0.25, 0.15))},
        prediction_timestamp="2026-09-05T10:30:00Z",
        git_sha="abc1234",
        min_train_matches=1,
    )

    validate_shadow_prediction_frame(predictions)
    assert set(predictions["model_name"]) == {"constant_home"}
    assert set(predictions["prediction_timestamp"]) == {"2026-09-05T10:30:00+00:00"}
    assert set(predictions["git_sha"]) == {"abc1234"}
    assert predictions["model_config"].str.contains("constant_home").all()
    assert predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_shadow_blend_predictions_include_config_and_valid_probabilities(
    historical_matches: pd.DataFrame,
    upcoming_fixtures: pd.DataFrame,
) -> None:
    predictions = generate_shadow_predictions(
        historical_matches,
        upcoming_fixtures,
        model_names=("blend_dc_fit_txg_50_50",),
        model_providers={
            "dc_fit_rho_each_fold": lambda: ConstantProbabilityModel("dc_fit_rho_each_fold", (0.7, 0.2, 0.1)),
            "txg_xg_pseudocount_010": lambda: ConstantProbabilityModel("txg_xg_pseudocount_010", (0.3, 0.3, 0.4)),
        },
        prediction_timestamp="2026-09-05T10:30:00Z",
        git_sha="abc1234",
        min_train_matches=1,
    )

    blend = predictions.loc[predictions["model_name"] == "blend_dc_fit_txg_50_50"]
    assert len(blend) == 2
    assert set(predictions["model_name"]) == {"blend_dc_fit_txg_50_50"}
    assert blend["model_config"].str.contains("dc_fit_rho_each_fold").all()
    assert blend[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_write_and_load_shadow_prediction_json(
    historical_matches: pd.DataFrame,
    upcoming_fixtures: pd.DataFrame,
    tmp_path: Path,
) -> None:
    predictions = generate_shadow_predictions(
        historical_matches,
        upcoming_fixtures.head(1),
        model_names=("constant_home",),
        model_providers={"constant_home": lambda: ConstantProbabilityModel("constant_home", (0.6, 0.25, 0.15))},
        prediction_timestamp="2026-09-05T10:30:00Z",
        git_sha="abc1234",
        min_train_matches=1,
    )
    path = tmp_path / "shadow_predictions.json"

    write_shadow_predictions(predictions, path, metadata={"purpose": "test"})
    loaded = load_shadow_predictions(path)

    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "shadow_predictions_v1"
    assert loaded.loc[0, "prediction_id"] == predictions.loc[0, "prediction_id"]


def test_fixture_loader_accepts_csv_and_generates_fallback_id(tmp_path: Path) -> None:
    path = tmp_path / "fixtures.csv"
    path.write_text("fixture_date,home_team,away_team\n2026-09-06,Arsenal,Chelsea\n", encoding="utf-8")

    fixtures = load_fixture_file(path)

    assert fixtures.loc[0, "fixture_id"] == "2026-09-06_arsenal_vs_chelsea"
    assert str(fixtures.loc[0, "match_date"].date()) == "2026-09-06"


def test_generate_script_validate_fixtures_only_does_not_require_output(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixtures.csv"
    fixture_path.write_text("fixture_date,home_team,away_team\n2026-09-06,Arsenal,Chelsea\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_shadow_predictions.py",
            "--fixtures",
            str(fixture_path),
            "--validate-fixtures-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Validated 1 fixture row(s)" in result.stdout
    assert "champion_dc_xg" in result.stdout


def test_replay_evaluator_handles_pending_fixtures(
    historical_matches: pd.DataFrame,
    upcoming_fixtures: pd.DataFrame,
) -> None:
    predictions = generate_shadow_predictions(
        historical_matches,
        upcoming_fixtures,
        model_names=("constant_home",),
        model_providers={"constant_home": lambda: ConstantProbabilityModel("constant_home", (0.6, 0.25, 0.15))},
        prediction_timestamp="2026-09-05T10:30:00Z",
        git_sha="abc1234",
        min_train_matches=1,
    )
    results = upcoming_fixtures.head(1).assign(home_goals=[2], away_goals=[1])

    replay = evaluate_shadow_predictions(predictions, results)

    assert replay["n_predictions"] == 2
    assert replay["n_completed"] == 1
    assert replay["n_pending"] == 1
    assert replay["metrics"][0]["model_name"] == "constant_home"


def test_replay_evaluator_computes_metrics_when_results_available(
    historical_matches: pd.DataFrame,
    upcoming_fixtures: pd.DataFrame,
) -> None:
    predictions = generate_shadow_predictions(
        historical_matches,
        upcoming_fixtures,
        model_names=("constant_home", "constant_away"),
        model_providers={
            "constant_home": lambda: ConstantProbabilityModel("constant_home", (0.7, 0.2, 0.1)),
            "constant_away": lambda: ConstantProbabilityModel("constant_away", (0.1, 0.2, 0.7)),
        },
        prediction_timestamp="2026-09-05T10:30:00Z",
        git_sha="abc1234",
        min_train_matches=1,
    )
    results = upcoming_fixtures.assign(actual_outcome=["H", "A"])

    replay = evaluate_shadow_predictions(predictions, results)

    assert replay["n_completed"] == 4
    assert {row["model_name"] for row in replay["metrics"]} == {"constant_home", "constant_away"}
    assert all("brier_score" in row and "log_loss" in row and "accuracy" in row for row in replay["metrics"])
