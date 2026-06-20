from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from evaluation.backtesting import BacktestConfig, build_rolling_folds, run_backtest_for_model


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
            "home_team": ["A", "B", "A", "B", "A", "B"],
            "away_team": ["B", "A", "B", "A", "B", "A"],
            "home_xg": [1.2, 1.1, 1.4, 1.0, 1.6, 1.3],
            "away_xg": [0.8, 1.5, 0.9, 1.4, 0.7, 1.2],
            "home_np_xg": [1.2, 1.1, 1.4, 1.0, 1.6, 1.3],
            "away_np_xg": [0.8, 1.5, 0.9, 1.4, 0.7, 1.2],
            "home_goals": [1, 1, 2, 0, 2, 1],
            "away_goals": [0, 2, 1, 1, 0, 1],
        },
        index=[10, 11, 12, 13, 14, 15],
    )


def test_fold_creation_is_deterministic(matches: pd.DataFrame) -> None:
    config = BacktestConfig(test_start="2025-09-22", test_end="2025-10-06", min_train_matches=2)

    first = [fold.metadata() for fold in build_rolling_folds(matches.sample(frac=1, random_state=7), config)]
    second = [fold.metadata() for fold in build_rolling_folds(matches.sample(frac=1, random_state=3), config)]

    assert first == second


def test_train_data_is_strictly_before_test_window(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-09-22", test_end="2025-10-06", min_train_matches=2),
    )

    for fold in folds:
        train_dates = matches.loc[list(fold.train_indices), "match_date"]
        assert (train_dates < fold.test_start).all()


def test_no_test_match_appears_in_train_fold(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-09-22", test_end="2025-10-06", min_train_matches=2),
    )

    for fold in folds:
        assert set(fold.train_indices).isdisjoint(fold.test_indices)


def test_round_labels_are_preserved_when_present(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-09-22", test_end="2025-09-29", min_train_matches=2),
    )

    assert folds[0].round_labels == ("R4",)
    assert folds[1].round_labels == ("R5",)


def test_round_label_filter_applies_to_test_rows_only(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(
            test_start="2025-09-22",
            test_end="2025-10-06",
            min_train_matches=2,
            round_labels=("R5",),
        ),
    )

    assert len(folds) == 1
    assert folds[0].round_labels == ("R5",)
    assert set(folds[0].train_indices) == {10, 11, 12, 13}
    assert set(folds[0].test_indices) == {14}


def test_insufficient_history_can_be_skipped_or_raised(matches: pd.DataFrame) -> None:
    skipped = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-09-08", test_end="2025-09-15", min_train_matches=4),
    )
    assert skipped == []

    with pytest.raises(ValueError, match="Insufficient training history"):
        build_rolling_folds(
            matches,
            BacktestConfig(
                test_start="2025-09-08",
                test_end="2025-09-15",
                min_train_matches=4,
                skip_insufficient_history=False,
            ),
        )


@dataclass
class FakeModel:
    @property
    def name(self) -> str:
        return "fake_model"

    @property
    def family(self) -> str:
        return "fake_family"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def spec(self) -> dict[str, Any]:
        return self.export_config()

    def fit(self, played: pd.DataFrame) -> FakeModel:
        self.train_size = len(played)
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "home_team": fixtures["home_team"].tolist(),
                "away_team": fixtures["away_team"].tolist(),
                "p_home_win": [0.4] * len(fixtures),
                "p_draw": [0.3] * len(fixtures),
                "p_away_win": [0.3] * len(fixtures),
                "trained_on": [self.train_size] * len(fixtures),
            }
        )

    def export_config(self) -> dict[str, Any]:
        return {"model_name": self.name, "model_family": self.family, "model_version": self.version}


def test_run_backtest_for_model_calls_fake_model(matches: pd.DataFrame) -> None:
    folds = build_rolling_folds(
        matches,
        BacktestConfig(test_start="2025-09-22", test_end="2025-10-06", min_train_matches=2),
    )

    result = run_backtest_for_model(FakeModel, matches, folds)

    assert result.model_name == "fake_model"
    assert len(result.folds) == 3
    assert len(result.predictions) == 3
    assert set(result.predictions["fold_id"]) == {"fold_001", "fold_002", "fold_003"}
    assert result.predictions.loc[0, "trained_on"] == 3

