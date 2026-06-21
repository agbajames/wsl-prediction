"""
Regularised attack/defence team-strength challenger.

This model is an offline evaluation challenger. It estimates simple smoothed
team attack and defence ratios from each training fold, then converts expected
goals into home/draw/away probabilities with an independent Poisson score grid.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = (
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)

SOURCE_COLUMNS = {
    "np_xg": ("home_np_xg", "away_np_xg"),
    "xg": ("home_xg", "away_xg"),
    "goals": ("home_goals", "away_goals"),
}


@dataclass
class RegularisedTeamStrengthModel:
    """Smoothed attack/defence Poisson challenger for small WSL samples."""

    strength_source: str = "np_xg"
    shrinkage_matches: float = 6.0
    max_goals: int = 8
    min_rate: float = 0.05
    max_rate: float = 5.0

    _resolved_source: str = field(default="unfitted", init=False)
    _home_rate: float = field(default=1.0, init=False)
    _away_rate: float = field(default=1.0, init=False)
    _league_attack_rate: float = field(default=1.0, init=False)
    _attack_strength: dict[str, float] = field(default_factory=dict, init=False)
    _defence_weakness: dict[str, float] = field(default_factory=dict, init=False)
    _training_matches: int = field(default=0, init=False)

    @property
    def name(self) -> str:
        return "regularised_team_strength"

    @property
    def family(self) -> str:
        return "regularised_attack_defence_poisson"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def spec(self) -> dict[str, Any]:
        return self.export_config()

    @property
    def attack_strengths(self) -> dict[str, float]:
        return dict(self._attack_strength)

    @property
    def defence_weaknesses(self) -> dict[str, float]:
        return dict(self._defence_weakness)

    def fit(self, played: pd.DataFrame) -> RegularisedTeamStrengthModel:
        """Fit smoothed team attack/defence strengths from training rows."""
        training = _played_with_results(played)
        home_source, away_source, resolved_source = _resolve_source_columns(training, self.strength_source)
        self._resolved_source = resolved_source
        self._training_matches = int(len(training))

        self._home_rate = _safe_mean(training[home_source], fallback=1.0)
        self._away_rate = _safe_mean(training[away_source], fallback=1.0)
        self._league_attack_rate = max(
            (float(training[home_source].sum()) + float(training[away_source].sum())) / (2.0 * len(training)),
            self.min_rate,
        )

        team_totals: dict[str, dict[str, float]] = {}
        for row in training.itertuples(index=False):
            home_team = str(row.home_team)
            away_team = str(row.away_team)
            home_value = float(getattr(row, home_source))
            away_value = float(getattr(row, away_source))
            _team_record(team_totals, home_team)["for"] += home_value
            _team_record(team_totals, home_team)["against"] += away_value
            _team_record(team_totals, home_team)["matches"] += 1.0
            _team_record(team_totals, away_team)["for"] += away_value
            _team_record(team_totals, away_team)["against"] += home_value
            _team_record(team_totals, away_team)["matches"] += 1.0

        for team, totals in team_totals.items():
            denominator = totals["matches"] + self.shrinkage_matches
            attack_rate = (totals["for"] + self.shrinkage_matches * self._league_attack_rate) / denominator
            defence_rate = (totals["against"] + self.shrinkage_matches * self._league_attack_rate) / denominator
            self._attack_strength[team] = _clip(attack_rate / self._league_attack_rate, 0.05, 20.0)
            self._defence_weakness[team] = _clip(defence_rate / self._league_attack_rate, 0.05, 20.0)

        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Predict fixture-level H/D/A probabilities."""
        if self._resolved_source == "unfitted":
            raise RuntimeError("RegularisedTeamStrengthModel must be fitted before predict().")

        rows = []
        for row in fixtures.itertuples(index=False):
            home_team = str(row.home_team)
            away_team = str(row.away_team)
            expected_home = self.expected_goals(home_team, away_team, venue="home")
            expected_away = self.expected_goals(away_team, home_team, venue="away")
            p_home, p_draw, p_away = win_draw_loss_probabilities(expected_home, expected_away, self.max_goals)
            rows.append(
                {
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "match_date": _match_date(row),
                    "round": _round_label(row),
                    "p_home_win": p_home,
                    "p_draw": p_draw,
                    "p_away_win": p_away,
                    "predicted_outcome": _predicted_outcome(p_home, p_draw, p_away),
                    "expected_home_goals": expected_home,
                    "expected_away_goals": expected_away,
                    "model_name": self.name,
                    "model_family": self.family,
                    "model_version": self.version,
                    "strength_source": self._resolved_source,
                    "training_matches": self._training_matches,
                }
            )
        return pd.DataFrame(rows)

    def expected_goals(self, attacking_team: str, defending_team: str, *, venue: str) -> float:
        """Return expected goals for an attacking team at home or away."""
        baseline = self._home_rate if venue == "home" else self._away_rate
        attack = self._attack_strength.get(attacking_team, 1.0)
        defence = self._defence_weakness.get(defending_team, 1.0)
        return _clip(baseline * attack * defence, self.min_rate, self.max_rate)

    def export_config(self) -> dict[str, Any]:
        config = asdict(self)
        for private_key in (
            "_resolved_source",
            "_home_rate",
            "_away_rate",
            "_league_attack_rate",
            "_attack_strength",
            "_defence_weakness",
            "_training_matches",
        ):
            config.pop(private_key, None)
        return {
            "model_name": self.name,
            "model_family": self.family,
            "model_version": self.version,
            "required_columns": list(REQUIRED_COLUMNS),
            "config": config,
            "resolved_strength_source": self._resolved_source,
        }


def win_draw_loss_probabilities(lambda_home: float, lambda_away: float, max_goals: int) -> tuple[float, float, float]:
    """Convert expected goals to H/D/A probabilities using a Poisson grid."""
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1.")
    home_pmf = [_poisson_pmf(score, lambda_home) for score in range(max_goals + 1)]
    away_pmf = [_poisson_pmf(score, lambda_away) for score in range(max_goals + 1)]
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for home_goals, home_prob in enumerate(home_pmf):
        for away_goals, away_prob in enumerate(away_pmf):
            probability = home_prob * away_prob
            if home_goals > away_goals:
                p_home += probability
            elif home_goals == away_goals:
                p_draw += probability
            else:
                p_away += probability
    return _normalise_probabilities(p_home, p_draw, p_away)


def _played_with_results(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required team-strength columns: {missing}")
    played = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()
    if played.empty:
        raise ValueError("Regularised team-strength model requires at least one played match.")
    if "match_date" in played.columns:
        played["match_date"] = pd.to_datetime(played["match_date"], format="ISO8601", errors="raise")
    return played


def _resolve_source_columns(df: pd.DataFrame, preferred_source: str) -> tuple[str, str, str]:
    sources = [preferred_source, "xg", "goals"]
    for source in sources:
        if source not in SOURCE_COLUMNS:
            continue
        home_column, away_column = SOURCE_COLUMNS[source]
        if {home_column, away_column}.issubset(df.columns) and df[[home_column, away_column]].notna().all().all():
            return home_column, away_column, source
    raise ValueError("No usable team-strength source found. Expected np_xg, xg, or goals columns.")


def _team_record(records: dict[str, dict[str, float]], team: str) -> dict[str, float]:
    if team not in records:
        records[team] = {"for": 0.0, "against": 0.0, "matches": 0.0}
    return records[team]


def _safe_mean(series: pd.Series, *, fallback: float) -> float:
    value = float(series.mean())
    if math.isnan(value) or value <= 0:
        return fallback
    return value


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def _normalise_probabilities(home: float, draw: float, away: float) -> tuple[float, float, float]:
    total = home + draw + away
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (home / total, draw / total, away / total)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _match_date(row: Any) -> str:
    value = getattr(row, "match_date", None)
    if value is None or pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date())


def _round_label(row: Any) -> str:
    value = getattr(row, "round_label", "")
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _predicted_outcome(p_home: float, p_draw: float, p_away: float) -> str:
    return ("H", "D", "A")[max(range(3), key=(p_home, p_draw, p_away).__getitem__)]
