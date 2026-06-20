"""
Reusable rolling backtest framework for evaluation model adapters.

The framework builds deterministic, date-based folds and runs any model that
implements the lightweight ``EvaluationModel`` protocol. Train rows are always
strictly before the test window to avoid future-data leakage.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from models.base import EvaluationModel


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for deterministic rolling backtest folds."""

    test_start: str | pd.Timestamp
    test_end: str | pd.Timestamp
    test_window_days: int = 7
    step_days: int = 7
    min_train_matches: int = 10
    train_start: str | pd.Timestamp | None = None
    date_column: str = "match_date"
    round_column: str = "round_label"
    round_labels: tuple[str, ...] | None = None
    skip_insufficient_history: bool = True


@dataclass(frozen=True)
class BacktestFold:
    """A single leakage-safe train/test split."""

    fold_id: str
    train_start: pd.Timestamp | None
    train_end_exclusive: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: tuple[Any, ...]
    test_indices: tuple[Any, ...]
    train_size: int
    test_size: int
    round_labels: tuple[str, ...] = field(default_factory=tuple)

    def metadata(self) -> dict[str, Any]:
        """Return deterministic, serializable fold metadata."""
        return {
            "fold_id": self.fold_id,
            "train_start": _date_or_none(self.train_start),
            "train_end_exclusive": self.train_end_exclusive.date().isoformat(),
            "test_start": self.test_start.date().isoformat(),
            "test_end": self.test_end.date().isoformat(),
            "train_size": self.train_size,
            "test_size": self.test_size,
            "round_labels": list(self.round_labels),
        }


@dataclass
class BacktestResult:
    """Predictions and metadata from running one model over reusable folds."""

    model_name: str
    model_family: str
    model_version: str
    config: dict[str, Any]
    folds: list[dict[str, Any]]
    predictions: pd.DataFrame
    skipped_folds: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-style result; predictions remain tabular as records."""
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "config": self.config,
            "folds": self.folds,
            "predictions": self.predictions.to_dict(orient="records"),
            "skipped_folds": self.skipped_folds,
        }


ModelProvider = EvaluationModel | Callable[[], EvaluationModel]


def build_rolling_folds(df: pd.DataFrame, config: BacktestConfig) -> list[BacktestFold]:
    """Build deterministic date-based folds from a match DataFrame.

    Each fold trains on rows with ``date_column < test_start`` and tests on
    rows in ``[test_start, test_end]``. If ``round_labels`` is provided and the
    round column exists, the test set is further filtered to those labels.
    """
    _validate_config(config)
    _validate_columns(df, config)

    working = df.copy()
    working[config.date_column] = pd.to_datetime(
        working[config.date_column],
        format="ISO8601",
        errors="raise",
    ).dt.normalize()
    working = working.sort_values([config.date_column, "home_team", "away_team"], kind="mergesort")

    test_start = pd.Timestamp(config.test_start).normalize()
    final_test_end = pd.Timestamp(config.test_end).normalize()
    train_start = pd.Timestamp(config.train_start).normalize() if config.train_start is not None else None

    folds: list[BacktestFold] = []
    current_start = test_start
    fold_number = 1

    while current_start <= final_test_end:
        current_end = min(current_start + pd.Timedelta(days=config.test_window_days - 1), final_test_end)

        train_mask = working[config.date_column] < current_start
        if train_start is not None:
            train_mask &= working[config.date_column] >= train_start

        test_mask = (working[config.date_column] >= current_start) & (working[config.date_column] <= current_end)
        if config.round_labels is not None and config.round_column in working.columns:
            test_mask &= working[config.round_column].astype(str).isin(config.round_labels)

        train = working.loc[train_mask]
        test = working.loc[test_mask]

        if len(train) >= config.min_train_matches and not test.empty:
            folds.append(
                BacktestFold(
                    fold_id=f"fold_{fold_number:03d}",
                    train_start=train[config.date_column].min() if not train.empty else train_start,
                    train_end_exclusive=current_start,
                    test_start=current_start,
                    test_end=current_end,
                    train_indices=tuple(train.index.tolist()),
                    test_indices=tuple(test.index.tolist()),
                    train_size=int(len(train)),
                    test_size=int(len(test)),
                    round_labels=_round_labels(test, config.round_column),
                )
            )
            fold_number += 1
        elif not config.skip_insufficient_history and not test.empty:
            raise ValueError(
                "Insufficient training history for "
                f"{current_start.date()} to {current_end.date()}: "
                f"{len(train)} rows found, {config.min_train_matches} required."
            )

        current_start += pd.Timedelta(days=config.step_days)

    return folds


def run_backtest_for_model(
    model_provider: ModelProvider,
    df: pd.DataFrame,
    folds: list[BacktestFold],
) -> BacktestResult:
    """Fit and predict one model over precomputed folds."""
    predictions: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, Any]] = []
    skipped_folds: list[dict[str, Any]] = []
    model_identity = _new_model(model_provider)

    for fold in folds:
        train = df.loc[list(fold.train_indices)].copy()
        test = df.loc[list(fold.test_indices)].copy()

        if train.empty or test.empty:
            skipped_folds.append({**fold.metadata(), "reason": "empty_train_or_test"})
            continue

        model = _new_model(model_provider)
        fitted = model.fit(train)
        fold_predictions = fitted.predict(test).copy()
        fold_predictions["fold_id"] = fold.fold_id
        fold_predictions["test_start"] = fold.test_start.date().isoformat()
        fold_predictions["test_end"] = fold.test_end.date().isoformat()
        predictions.append(fold_predictions)
        fold_metadata.append(fold.metadata())

    all_predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return BacktestResult(
        model_name=model_identity.name,
        model_family=model_identity.family,
        model_version=model_identity.version,
        config=model_identity.export_config(),
        folds=fold_metadata,
        predictions=all_predictions,
        skipped_folds=skipped_folds,
    )


def _validate_config(config: BacktestConfig) -> None:
    if config.test_window_days < 1:
        raise ValueError("test_window_days must be at least 1.")
    if config.step_days < 1:
        raise ValueError("step_days must be at least 1.")
    if config.min_train_matches < 0:
        raise ValueError("min_train_matches cannot be negative.")
    if pd.Timestamp(config.test_start) > pd.Timestamp(config.test_end):
        raise ValueError("test_start must be on or before test_end.")


def _validate_columns(df: pd.DataFrame, config: BacktestConfig) -> None:
    if config.date_column not in df.columns:
        raise ValueError(f"DataFrame missing date column {config.date_column!r}.")
    for column in ("home_team", "away_team"):
        if column not in df.columns:
            raise ValueError(f"DataFrame missing required ordering column {column!r}.")


def _round_labels(test: pd.DataFrame, round_column: str) -> tuple[str, ...]:
    if round_column not in test.columns:
        return ()
    labels = test[round_column].dropna().astype(str).unique().tolist()
    return tuple(sorted(labels))


def _new_model(model_provider: ModelProvider) -> EvaluationModel:
    if callable(model_provider):
        return model_provider()
    return deepcopy(model_provider)


def _date_or_none(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def config_to_dict(config: BacktestConfig) -> dict[str, Any]:
    """Serialize a backtest config for scripts and logs."""
    result = asdict(config)
    result["test_start"] = pd.Timestamp(config.test_start).date().isoformat()
    result["test_end"] = pd.Timestamp(config.test_end).date().isoformat()
    result["train_start"] = _date_or_none(pd.Timestamp(config.train_start) if config.train_start is not None else None)
    result["round_labels"] = list(config.round_labels) if config.round_labels is not None else None
    return result
