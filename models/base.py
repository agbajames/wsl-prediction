"""
Common interface for evaluation model adapters.

The interface is intentionally small: it gives future evaluation runners a
stable way to inspect model identity/configuration, fit on historical matches,
and produce predictions for fixtures without knowing implementation details.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class EvaluationModel(Protocol):
    """Minimal protocol implemented by models that can be evaluated."""

    @property
    def name(self) -> str:
        """Stable model name used in evaluation logs."""

    @property
    def family(self) -> str:
        """Model family or modelling approach."""

    @property
    def version(self) -> str:
        """Frozen model/specification version."""

    @property
    def spec(self) -> dict[str, Any]:
        """Serializable model identity and configuration metadata."""

    def fit(self, played: pd.DataFrame) -> EvaluationModel:
        """Fit the model on played matches and return self."""

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Return fixture-level predictions in an evaluation-friendly shape."""

    def export_config(self) -> dict[str, Any]:
        """Return serializable configuration and metadata."""

