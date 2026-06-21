"""
Lightweight model registry for local evaluation experiments.
"""

from __future__ import annotations

from collections.abc import Callable

from models.base import EvaluationModel
from models.baselines import EloBaseline, NaiveOutcomeRateBaseline
from models.champion_dc_xg import ChampionDCXGModel
from models.logistic import ImprovedLogisticRegressionChallenger, LogisticRegressionChallenger
from models.neural_network import NeuralNetworkChallenger
from models.poisson_regression import PoissonRegressionChallenger
from models.team_strength import RegularisedTeamStrengthModel
from models.tree_based import RandomForestChallenger

ModelConstructor = Callable[[], EvaluationModel]

MODEL_REGISTRY: dict[str, ModelConstructor] = {
    "champion_dc_xg": ChampionDCXGModel,
    "naive_outcome_rate": NaiveOutcomeRateBaseline,
    "elo_baseline": EloBaseline,
    "logistic_regression": LogisticRegressionChallenger,
    "improved_logistic_regression": ImprovedLogisticRegressionChallenger,
    "neural_network": NeuralNetworkChallenger,
    "regularised_team_strength": RegularisedTeamStrengthModel,
    "poisson_regression": PoissonRegressionChallenger,
    "random_forest": RandomForestChallenger,
}


def available_models() -> list[str]:
    """Return registered model names in deterministic order."""
    return sorted(MODEL_REGISTRY)


def get_model_constructor(model_name: str) -> ModelConstructor:
    """Return a model constructor or raise a clear error."""
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {model_name!r}. Available models: {available_models()}"
        ) from exc


def get_model_constructors(model_names: list[str] | tuple[str, ...]) -> dict[str, ModelConstructor]:
    """Return constructors keyed by requested model name."""
    return {model_name: get_model_constructor(model_name) for model_name in model_names}

