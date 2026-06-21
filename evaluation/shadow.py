"""
Shadow prediction artefact helpers for pre-match model tracking.

The functions in this module are evaluation-only. They fit selected candidate
models on historical rows available before each fixture date, write timestamped
prediction artefacts, and replay those artefacts after actual results arrive.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.blending import build_default_component_providers, get_blend_spec, normalise_probability_frame
from evaluation.compare import PROBABILITY_COLUMNS, compare_model_results
from experiments.registry import get_model_constructor
from models.base import EvaluationModel

DEFAULT_SHADOW_MODELS: tuple[str, ...] = (
    "champion_dc_xg",
    "dc_fit_rho_each_fold",
    "txg_xg_pseudocount_010",
    "blend_dc_fit_txg_50_50",
    "regularised_team_strength",
    "improved_logistic_regression",
    "random_forest",
)
REQUIRED_FIXTURE_COLUMNS = ("fixture_date", "home_team", "away_team")
SHADOW_PREDICTION_COLUMNS = (
    "prediction_id",
    "prediction_timestamp",
    "git_sha",
    "model_name",
    "model_family",
    "model_version",
    "model_config",
    "fixture_id",
    "fixture_date",
    "home_team",
    "away_team",
    "p_home_win",
    "p_draw",
    "p_away_win",
    "predicted_outcome",
)

ModelProvider = Callable[[], EvaluationModel]


def load_fixture_file(path: Path) -> pd.DataFrame:
    """Load upcoming fixtures from CSV or JSON and validate the shadow schema."""
    if path.suffix.lower() == ".csv":
        fixtures = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        fixtures = pd.DataFrame(payload["fixtures"] if isinstance(payload, dict) and "fixtures" in payload else payload)
    else:
        raise ValueError(f"Unsupported fixture file extension {path.suffix!r}. Use CSV or JSON.")
    return normalise_fixture_frame(fixtures)


def load_shadow_predictions(path: Path) -> pd.DataFrame:
    """Load saved shadow prediction artefacts from CSV or JSON."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["predictions"] if isinstance(payload, dict) and "predictions" in payload else payload
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported prediction file extension {path.suffix!r}. Use CSV or JSON.")


def write_shadow_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write shadow predictions as JSON or CSV without changing production state."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        predictions.to_csv(output_path, index=False)
        return
    if output_path.suffix.lower() != ".json":
        raise ValueError("Shadow prediction output must end in .json or .csv.")
    payload = {
        "metadata": metadata or {},
        "schema_version": "shadow_predictions_v1",
        "predictions": predictions.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(payload, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")


def normalise_fixture_frame(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Return fixtures with stable IDs, date columns and required fields."""
    normalized = fixtures.copy()
    if "fixture_date" not in normalized.columns and "match_date" in normalized.columns:
        normalized["fixture_date"] = normalized["match_date"]
    missing = [column for column in REQUIRED_FIXTURE_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"Fixture input missing required columns: {missing}")

    normalized["fixture_date"] = pd.to_datetime(normalized["fixture_date"], format="ISO8601", errors="raise").dt.normalize()
    normalized["match_date"] = normalized["fixture_date"]
    normalized["home_team"] = normalized["home_team"].astype(str).str.strip()
    normalized["away_team"] = normalized["away_team"].astype(str).str.strip()
    if "fixture_id" not in normalized.columns:
        normalized["fixture_id"] = [
            _fallback_fixture_id(row.fixture_date, row.home_team, row.away_team)
            for row in normalized.loc[:, ["fixture_date", "home_team", "away_team"]].itertuples(index=False)
        ]
    else:
        normalized["fixture_id"] = normalized["fixture_id"].fillna("").astype(str)
        missing_ids = normalized["fixture_id"].str.strip() == ""
        normalized.loc[missing_ids, "fixture_id"] = [
            _fallback_fixture_id(row.fixture_date, row.home_team, row.away_team)
            for row in normalized.loc[missing_ids, ["fixture_date", "home_team", "away_team"]].itertuples(index=False)
        ]
    return normalized.sort_values(["fixture_date", "home_team", "away_team"], kind="mergesort").reset_index(drop=True)


def build_shadow_model_providers(model_names: Sequence[str]) -> dict[str, ModelProvider]:
    """Return callable providers for non-blend shadow models."""
    default_components = build_default_component_providers()
    providers: dict[str, ModelProvider] = {}
    for model_name in model_names:
        if model_name.startswith("blend_"):
            continue
        if model_name in default_components:
            providers[model_name] = default_components[model_name]
        else:
            providers[model_name] = get_model_constructor(model_name)
    return providers


def validate_shadow_model_names(model_names: Sequence[str]) -> tuple[str, ...]:
    """Validate that requested shadow models and blend components are callable."""
    requested = tuple(dict.fromkeys(model_names))
    required_model_names = _required_model_names(requested)
    build_shadow_model_providers(required_model_names)
    return requested


def generate_shadow_predictions(
    historical_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    model_names: Sequence[str] = DEFAULT_SHADOW_MODELS,
    prediction_timestamp: str | pd.Timestamp | None = None,
    git_sha: str | None = None,
    min_train_matches: int = 10,
    model_providers: dict[str, ModelProvider] | None = None,
) -> pd.DataFrame:
    """Generate timestamped pre-match predictions for candidate models."""
    if min_train_matches < 0:
        raise ValueError("min_train_matches cannot be negative.")

    played = historical_matches.copy()
    if "match_date" not in played.columns:
        raise ValueError("Historical match data must include match_date.")
    played["match_date"] = pd.to_datetime(played["match_date"], format="ISO8601", errors="raise").dt.normalize()
    fixtures = normalise_fixture_frame(fixtures)
    timestamp = _timestamp_iso(prediction_timestamp)
    sha = git_sha or current_git_sha()
    requested = tuple(dict.fromkeys(model_names))
    blend_names = [name for name in requested if name.startswith("blend_")]
    required_model_names = _required_model_names(requested)
    providers = model_providers or build_shadow_model_providers(required_model_names)

    rows: list[pd.DataFrame] = []
    for fixture_date, fixture_group in fixtures.groupby("fixture_date", sort=True):
        train = played.loc[played["match_date"] < fixture_date].copy()
        if len(train) < min_train_matches:
            raise ValueError(
                f"Only {len(train)} historical rows are available before {fixture_date.date()}; "
                f"{min_train_matches} required."
            )
        prediction_frames = _predict_base_models(
            train,
            fixture_group.copy(),
            providers={name: providers[name] for name in required_model_names if name in providers},
            timestamp=timestamp,
            git_sha=sha,
        )
        for blend_name in blend_names:
            prediction_frames[blend_name] = _predict_blend(blend_name, prediction_frames)
            prediction_frames[blend_name] = _stamp_common_metadata(
                prediction_frames[blend_name],
                fixtures=fixture_group,
                prediction_timestamp=timestamp,
                git_sha=sha,
            )
        rows.extend(prediction_frames[name] for name in requested if name in prediction_frames)

    if not rows:
        return pd.DataFrame(columns=SHADOW_PREDICTION_COLUMNS)
    predictions = pd.concat(rows, ignore_index=True)
    predictions = predictions.reindex(columns=[*SHADOW_PREDICTION_COLUMNS, *[c for c in predictions.columns if c not in SHADOW_PREDICTION_COLUMNS]])
    validate_shadow_prediction_frame(predictions)
    return predictions


def evaluate_shadow_predictions(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
) -> dict[str, Any]:
    """Replay saved shadow predictions against completed results."""
    prediction_rows = predictions.copy()
    result_rows = normalise_result_frame(results)
    merged = _merge_results(prediction_rows, result_rows)
    completed = merged.loc[merged["actual_outcome"].notna()].copy()
    pending = merged.loc[merged["actual_outcome"].isna()].copy()
    comparison = compare_model_results(completed) if not completed.empty else pd.DataFrame()
    return {
        "n_predictions": int(len(merged)),
        "n_completed": int(len(completed)),
        "n_pending": int(len(pending)),
        "completed_predictions": completed.to_dict(orient="records"),
        "pending_predictions": pending.to_dict(orient="records"),
        "metrics": comparison.to_dict(orient="records"),
    }


def normalise_result_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Return actual-result rows keyed for replay evaluation."""
    normalized = results.copy()
    if "fixture_date" not in normalized.columns and "match_date" in normalized.columns:
        normalized["fixture_date"] = normalized["match_date"]
    if "actual_outcome" not in normalized.columns:
        if {"home_goals", "away_goals"}.issubset(normalized.columns):
            normalized["actual_outcome"] = [
                _outcome_from_goals(home, away)
                for home, away in normalized.loc[:, ["home_goals", "away_goals"]].itertuples(index=False)
            ]
        else:
            raise ValueError("Results must include actual_outcome or home_goals/away_goals.")
    fixtures = normalise_fixture_frame(normalized)
    fixtures["actual_outcome"] = normalized["actual_outcome"].map(_normalise_outcome)
    return fixtures


def validate_shadow_prediction_frame(predictions: pd.DataFrame) -> None:
    """Validate required schema and probability discipline for shadow artefacts."""
    missing = [column for column in SHADOW_PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f"Shadow prediction artefact missing required columns: {missing}")
    required_non_empty = (
        "prediction_timestamp",
        "git_sha",
        "model_name",
        "model_version",
        "model_config",
        "fixture_id",
        "fixture_date",
        "home_team",
        "away_team",
    )
    for column in required_non_empty:
        if predictions[column].isna().any() or (predictions[column].astype(str).str.strip() == "").any():
            raise ValueError(f"Shadow prediction column {column!r} must be present for every row.")
    probabilities = predictions.loc[:, PROBABILITY_COLUMNS].astype(float)
    if (probabilities < 0).any().any() or (probabilities > 1).any().any():
        raise ValueError("Shadow prediction probabilities must be in [0, 1].")
    row_sums = probabilities.sum(axis=1)
    if not ((row_sums - 1.0).abs() <= 1e-6).all():
        raise ValueError("Shadow prediction probabilities must sum to approximately 1.")


def current_git_sha() -> str:
    """Return the current Git SHA, or unknown when Git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _predict_base_models(
    train: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    providers: dict[str, ModelProvider],
    timestamp: str,
    git_sha: str,
) -> dict[str, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}
    for requested_name, provider in providers.items():
        model = provider()
        fitted = model.fit(train)
        raw = fitted.predict(fixtures).copy()
        normalized = normalise_probability_frame(raw)
        normalized["model_name"] = getattr(fitted, "name", requested_name)
        normalized["model_family"] = getattr(fitted, "family", "")
        normalized["model_version"] = getattr(fitted, "version", "")
        normalized["model_config"] = json.dumps(fitted.export_config(), default=_json_default, sort_keys=True)
        predictions[requested_name] = _stamp_common_metadata(
            normalized,
            fixtures=fixtures,
            prediction_timestamp=timestamp,
            git_sha=git_sha,
        )
    return predictions


def _required_model_names(model_names: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for model_name in model_names:
        if not model_name.startswith("blend_"):
            names.append(model_name)
            continue
        names.extend(get_blend_spec(model_name).components)
    return tuple(dict.fromkeys(names))


def _predict_blend(blend_name: str, predictions_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    spec = get_blend_spec(blend_name)
    missing = [component for component in spec.components if component not in predictions_by_model]
    if missing:
        raise ValueError(f"Blend {blend_name!r} requires component predictions that were not generated: {missing}")
    frames = [predictions_by_model[component].copy() for component in spec.components]
    probabilities = sum(
        weight * frame.loc[:, PROBABILITY_COLUMNS].astype(float)
        for weight, frame in zip(_normalised_weights(spec.weights), frames)
    )
    template = frames[0].copy()
    template.loc[:, PROBABILITY_COLUMNS] = probabilities.div(probabilities.sum(axis=1), axis=0)
    template["model_name"] = spec.model_name
    template["model_family"] = spec.model_family
    template["model_version"] = spec.model_version
    template["model_config"] = json.dumps(
        {
            "model_name": spec.model_name,
            "model_family": spec.model_family,
            "model_version": spec.model_version,
            "components": list(spec.components),
            "weights": list(spec.weights),
            "offline_only": True,
        },
        sort_keys=True,
    )
    template["predicted_outcome"] = _predicted_outcomes(template)
    return template


def _stamp_common_metadata(
    predictions: pd.DataFrame,
    *,
    fixtures: pd.DataFrame,
    prediction_timestamp: str,
    git_sha: str,
) -> pd.DataFrame:
    stamped = predictions.reset_index(drop=True).copy()
    fixture_meta = fixtures.reset_index(drop=True).copy()
    stamped["fixture_id"] = fixture_meta["fixture_id"].astype(str)
    stamped["fixture_date"] = fixture_meta["fixture_date"].dt.date.astype(str)
    stamped["match_date"] = stamped["fixture_date"]
    stamped["home_team"] = fixture_meta["home_team"].astype(str)
    stamped["away_team"] = fixture_meta["away_team"].astype(str)
    stamped["prediction_timestamp"] = prediction_timestamp
    stamped["git_sha"] = git_sha
    if "model_config" not in stamped.columns:
        stamped["model_config"] = "{}"
    stamped["predicted_outcome"] = _predicted_outcomes(stamped)
    stamped["prediction_id"] = [
        f"{model_name}|{fixture_id}|{prediction_timestamp}"
        for model_name, fixture_id in stamped.loc[:, ["model_name", "fixture_id"]].itertuples(index=False)
    ]
    return stamped


def _merge_results(predictions: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["fixture_id"] if "fixture_id" in predictions.columns and "fixture_id" in results.columns else [
        "fixture_date",
        "home_team",
        "away_team",
    ]
    right = results.loc[:, [*merge_keys, "actual_outcome"]].drop_duplicates()
    merged = predictions.merge(right, how="left", on=merge_keys)
    return merged


def _fallback_fixture_id(fixture_date: pd.Timestamp, home_team: str, away_team: str) -> str:
    home = str(home_team).strip().lower().replace(" ", "-")
    away = str(away_team).strip().lower().replace(" ", "-")
    return f"{pd.Timestamp(fixture_date).date().isoformat()}_{home}_vs_{away}"


def _timestamp_iso(value: str | pd.Timestamp | None) -> str:
    timestamp = pd.Timestamp.utcnow() if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _predicted_outcomes(predictions: pd.DataFrame) -> list[str]:
    labels = {"p_home_win": "H", "p_draw": "D", "p_away_win": "A"}
    return [labels[column] for column in predictions.loc[:, PROBABILITY_COLUMNS].astype(float).idxmax(axis=1)]


def _normalised_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
    total = float(sum(weights))
    return tuple(float(weight) / total for weight in weights)


def _outcome_from_goals(home_goals: Any, away_goals: Any) -> str | None:
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    home = int(home_goals)
    away = int(away_goals)
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"


def _normalise_outcome(value: Any) -> str | None:
    if pd.isna(value):
        return None
    label = str(value).strip().upper()
    aliases = {
        "HOME": "H",
        "HOME_WIN": "H",
        "H": "H",
        "DRAW": "D",
        "D": "D",
        "AWAY": "A",
        "AWAY_WIN": "A",
        "A": "A",
    }
    if label not in aliases:
        raise ValueError(f"Invalid actual outcome: {value!r}. Expected H, D, or A.")
    return aliases[label]


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
