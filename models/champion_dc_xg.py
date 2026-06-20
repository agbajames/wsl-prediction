"""
Champion adapter for the current xG-driven Dixon-Coles/Poisson model.

This module adapts the existing production model implementation in
``model.wsl_xg_model`` to the common evaluation interface. It intentionally
delegates fitting and prediction to that module instead of reimplementing model
logic here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from model.wsl_xg_model import (
    ModelConfig,
    estimate_penalty_rates,
    estimate_team_strengths,
    fit_rho,
    predict_fixtures,
    split_played_future,
)

MODEL_NAME = "champion_dc_xg"
MODEL_FAMILY = "xg_dixon_coles_poisson"
MODEL_VERSION = "v1"
REQUIRED_INPUT_SCHEMA = "data.schemas.CHAMPION_REQUIRED_COLUMNS"


@dataclass
class ChampionDCXGModel:
    """Evaluation adapter for the frozen champion model specification."""

    config: ModelConfig = field(default_factory=ModelConfig)
    fit_rho_when_config_none: bool = True

    _strengths: Any | None = field(default=None, init=False, repr=False)
    _home_pen_rates: dict[str, float] | None = field(default=None, init=False, repr=False)
    _away_pen_rates: dict[str, float] | None = field(default=None, init=False, repr=False)
    _rho: float | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> ChampionDCXGModel:
        """Build the adapter from a serializable config dictionary."""
        if config is None:
            return cls()

        raw_config = config.get("config", config)
        model_config = ModelConfig(
            alpha=float(raw_config.get("alpha", ModelConfig.alpha)),
            decay_half_life_days=float(
                raw_config.get(
                    "decay_half_life_days",
                    raw_config.get("decay_days", ModelConfig.decay_half_life_days),
                )
            ),
            rho=raw_config.get("rho", ModelConfig.rho),
            max_goals=int(raw_config.get("max_goals", ModelConfig.max_goals)),
            league_avg_goals=float(raw_config.get("league_avg_goals", ModelConfig.league_avg_goals)),
            home_advantage_prior=float(raw_config.get("home_advantage_prior", ModelConfig.home_advantage_prior)),
            penalty_shrinkage_n=float(raw_config.get("penalty_shrinkage_n", ModelConfig.penalty_shrinkage_n)),
            xg_pseudocount=float(raw_config.get("xg_pseudocount", ModelConfig.xg_pseudocount)),
            bootstrap_n=int(raw_config.get("bootstrap_n", ModelConfig.bootstrap_n)),
            rho_grid=tuple(raw_config.get("rho_grid", ModelConfig.rho_grid)),
        )
        rho_behavior = raw_config.get("rho_behavior", config.get("rho_behavior", "fixed_default"))
        return cls(
            config=model_config,
            fit_rho_when_config_none=rho_behavior in {"fit_when_none", "fit"},
        )

    @property
    def name(self) -> str:
        return MODEL_NAME

    @property
    def family(self) -> str:
        return MODEL_FAMILY

    @property
    def version(self) -> str:
        return MODEL_VERSION

    @property
    def spec(self) -> dict[str, Any]:
        return self.export_config()

    @property
    def rho(self) -> float | None:
        return self._rho

    def fit(self, played: pd.DataFrame) -> ChampionDCXGModel:
        """Fit the champion using the existing production model functions."""
        self._strengths = estimate_team_strengths(played, self.config)
        self._home_pen_rates, self._away_pen_rates = estimate_penalty_rates(played, self.config)
        self._rho = self._resolve_rho(played)
        return self

    def fit_from_dataset(
        self,
        df: pd.DataFrame,
        *,
        train_before: pd.Timestamp,
        predict_from: pd.Timestamp,
        predict_to: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a full dataset with existing champion logic, then fit."""
        played, future = split_played_future(df, train_before, predict_from, predict_to)
        self.fit(played)
        return played, future

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Predict fixtures and return a tabular, evaluation-friendly output."""
        if (
            self._strengths is None
            or self._home_pen_rates is None
            or self._away_pen_rates is None
            or self._rho is None
        ):
            raise RuntimeError("ChampionDCXGModel must be fitted before predict().")

        predictions = predict_fixtures(
            fixtures,
            self._strengths,
            self._home_pen_rates,
            self._away_pen_rates,
            self.config,
            rho=self._rho,
        )
        rows = []
        for prediction in predictions:
            row = prediction.to_dict()
            row.update(
                {
                    "model_name": self.name,
                    "model_family": self.family,
                    "model_version": self.version,
                    "rho": self._rho,
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def export_config(self) -> dict[str, Any]:
        """Return frozen identity and current champion configuration."""
        config = asdict(self.config)
        config["rho_grid"] = list(config["rho_grid"])
        return {
            "model_name": self.name,
            "model_family": self.family,
            "model_version": self.version,
            "required_input_schema": REQUIRED_INPUT_SCHEMA,
            "config": config,
            "rho_behavior": "fit_when_none" if self.fit_rho_when_config_none else "fixed_default",
        }

    def _resolve_rho(self, played: pd.DataFrame) -> float:
        if self.config.rho is not None:
            return self.config.rho
        if self.fit_rho_when_config_none:
            return fit_rho(played, self._strengths, self.config)
        return -0.13
