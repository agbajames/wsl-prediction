from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from evaluation.backtesting import BacktestConfig
from evaluation.blending import (
    BlendSpec,
    available_blend_names,
    blend_prediction_frames,
    build_default_component_providers,
    get_blend_spec,
    normalise_probability_frame,
    run_non_market_blending_experiment,
)
from scripts.run_non_market_blending_experiment import main


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


def prediction_frame(model_name: str, probabilities: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Man City"],
            "match_date": ["2025-10-01", "2025-10-08"],
            "fold_id": ["fold_001", "fold_002"],
            "p_home_win": [row[0] for row in probabilities],
            "p_draw": [row[1] for row in probabilities],
            "p_away_win": [row[2] for row in probabilities],
            "model_name": model_name,
            "actual_outcome": ["H", "A"],
            "home_goals": [2, 0],
            "away_goals": [0, 1],
        }
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


def test_blend_registry_exposes_small_predeclared_grid() -> None:
    names = available_blend_names()

    assert names == [
        "blend_champion_regularised_50_50",
        "blend_champion_regularised_70_30",
        "blend_champion_improved_logistic_70_30",
        "blend_champion_random_forest_70_30",
        "blend_champion_regularised_improved_logistic_60_20_20",
        "blend_dc_fit_txg_50_50",
    ]
    assert get_blend_spec("blend_champion_regularised_70_30").weights == (0.70, 0.30)
    with pytest.raises(ValueError, match="Unknown blend"):
        get_blend_spec("not_a_blend")


def test_default_components_keep_champion_reference_provider() -> None:
    providers = build_default_component_providers()

    assert providers["champion_dc_xg"]().name == "champion_dc_xg"
    assert "dc_fit_rho_each_fold" in providers
    assert "txg_xg_pseudocount_010" in providers
    assert "neural_network" in providers


def test_normalise_probability_frame_handles_percentage_rows() -> None:
    raw = prediction_frame("champion_dc_xg", [(55.0, 25.0, 20.0), (42.0, 28.0, 30.0)])

    normalized = normalise_probability_frame(raw)

    assert normalized[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert normalized.loc[0, "p_home_win"] == pytest.approx(0.55)


def test_fixed_weight_blend_probabilities_are_valid() -> None:
    spec = BlendSpec(
        model_name="blend_test",
        components=("a", "b"),
        weights=(0.70, 0.30),
        description="test blend",
    )
    blended = blend_prediction_frames(
        {
            "a": prediction_frame("a", [(0.60, 0.20, 0.20), (60.0, 20.0, 20.0)]),
            "b": prediction_frame("b", [(0.30, 0.30, 0.40), (0.20, 0.30, 0.50)]),
        },
        spec,
    )

    assert blended["model_name"].tolist() == ["blend_test", "blend_test"]
    assert blended[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert (blended[["p_home_win", "p_draw", "p_away_win"]] >= 0).all().all()
    assert blended.loc[0, "p_home_win"] == pytest.approx(0.51)
    assert blended.loc[0, "predicted_outcome"] == "H"


def test_blend_rejects_invalid_inputs() -> None:
    spec = BlendSpec("bad", ("a", "b"), (0.5, -0.5), "bad")

    with pytest.raises(ValueError, match="negative weights"):
        blend_prediction_frames({"a": prediction_frame("a", [(0.3, 0.3, 0.4), (0.4, 0.3, 0.3)])}, spec)

    missing = BlendSpec("missing", ("a", "b"), (0.5, 0.5), "missing")
    with pytest.raises(ValueError, match="missing component predictions"):
        blend_prediction_frames({"a": prediction_frame("a", [(0.3, 0.3, 0.4), (0.4, 0.3, 0.3)])}, missing)


def test_blend_rejects_misaligned_component_frames() -> None:
    spec = BlendSpec("misaligned", ("a", "b"), (0.5, 0.5), "misaligned")
    left = prediction_frame("a", [(0.3, 0.3, 0.4), (0.4, 0.3, 0.3)])
    right = prediction_frame("b", [(0.3, 0.3, 0.4), (0.4, 0.3, 0.3)])
    right.loc[1, "away_team"] = "Arsenal"

    with pytest.raises(ValueError, match="not aligned"):
        blend_prediction_frames({"a": left, "b": right}, spec)


def test_non_market_blending_experiment_runs_shared_comparison(matches: pd.DataFrame) -> None:
    payload = run_non_market_blending_experiment(
        matches,
        backtest_config=BacktestConfig(test_start="2025-09-22", test_end="2025-10-27", min_train_matches=3),
        component_providers={
            "model_a": lambda: ConstantProbabilityModel("model_a", (0.55, 0.25, 0.20)),
            "model_b": lambda: ConstantProbabilityModel("model_b", (0.30, 0.30, 0.40)),
        },
        blend_specs=(BlendSpec("blend_model_a_b_70_30", ("model_a", "model_b"), (0.70, 0.30), "test"),),
        n_bins=4,
        top_n=2,
    )

    assert payload["components"] == ["model_a", "model_b"]
    assert payload["blends"][0]["model_name"] == "blend_model_a_b_70_30"
    assert {row["model_name"] for row in payload["prediction_rows"]} == {
        "model_a",
        "model_b",
        "blend_model_a_b_70_30",
    }
    assert "# Non-Market Blending Experiment" in payload["markdown"]


def test_cli_lists_available_blends(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.argv", ["run_non_market_blending_experiment.py", "--csv", "unused.csv", "--test-start", "2025-10-01", "--test-end", "2025-10-08", "--output-md", "unused.md", "--list-blends"])

    main()

    assert "blend_champion_regularised_50_50" in capsys.readouterr().out


def test_cli_writes_report_artifacts_with_selected_real_blend(
    matches: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "matches.csv"
    md_path = tmp_path / "non_market_blending.md"
    json_path = tmp_path / "non_market_blending.json"
    matches.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_non_market_blending_experiment.py",
            "--csv",
            str(csv_path),
            "--blend",
            "blend_champion_regularised_50_50",
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
    assert "blend_champion_regularised_50_50" in md_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["blends"][0]["model_name"] == (
        "blend_champion_regularised_50_50"
    )
