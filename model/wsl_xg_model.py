
#!/usr/bin/env python3
"""
WSL xG-driven match prediction model (v3).

A modified Dixon-Coles framework estimating team attack/defence strengths from
match-level np_xG via weighted ridge regression, then generating scoreline
probabilities from expected goals (λs).

Key differences from canonical Dixon-Coles:
    - Team strengths estimated via regularised log-linear regression on np_xG
      (assumes log-normal noise around expected goals) rather than MLE on
      observed goals (Poisson likelihood). This trades strict distributional
      assumptions for better signal in small-sample leagues like the WSL.
    - ρ (low-score correlation) is fitted via profile log-likelihood grid
      search on observed goals, conditional on fixed xG-based strengths.
    - Penalty xG is modelled as an independent Poisson component, convolved
      with the DC-corrected open-play scoreline matrix. This ensures the DC
      adjustment only captures open-play tactical correlation.

Additional features:
    - Time-decay weighting (recent matches matter more)
    - Walk-forward backtesting with Brier score, log-loss, and calibration
    - Bootstrap confidence intervals on match predictions
    - Structured logging throughout

Splitting is now entirely DATE-BASED — round_label is preserved in the data
for convenience but is not used for train/test splits. This handles postponed
and rearranged fixtures naturally: a match's match_date reflects when it was
actually played, regardless of which round_label it carries.

Usage:
    # Predict the R20 batch (25/04 weekend + 29/04 rearranged R14)
    python wsl_xg_model.py --csv raw/wsl_round_22.csv \
        --train-before 2026-05-13 \
        --predict-from 2026-05-13 \
        --predict-to 2026-05-17 \
        --fit-rho \
        --out-predictions predictions/predictions_round_22.csv

    # With walk-forward backtest (weekly batches)
    python wsl_xg_model.py --csv raw/wsl_round_21.csv \
        --train-before 2026-05-02 \
        --predict-from 2026-05-02 --predict-to 2026-05-06 \
        --backtest --backtest-start 2025-10-01
"""
from __future__ import annotations

import argparse
import heapq
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("wsl_xg_model")


def configure_logging(verbosity: int = 1) -> None:
    """Set up structured logging. verbosity: 0=WARNING, 1=INFO, 2=DEBUG."""
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        verbosity, logging.DEBUG
    )
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.setLevel(level)
    logger.addHandler(handler)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Model hyperparameters.

    Attributes:
        alpha: Ridge regularisation strength (higher = more shrinkage toward
            league average).  Applied to team attack/defence parameters.
        decay_half_life_days: Exponential time-decay half-life. A match played
            ``decay_half_life_days`` ago receives half the weight of the most
            recent match.
        rho: Dixon-Coles low-score correlation.  Negative values increase draw
            probability and reduce 1-0 / 0-1 probability, matching empirical
            football patterns.  Set to None to fit from data.
        max_goals: Upper bound of the scoreline grid.  Poisson tail beyond this
            is renormalised away.
        league_avg_goals: Prior for league-average goals per team per match.
            Used only as documentation / reference; the intercept is estimated.
        home_advantage_prior: Soft prior strength on home advantage (controls
            regularisation weight on that coefficient).
        penalty_shrinkage_n: Pseudo-sample-size for James-Stein shrinkage of
            per-team penalty rates.  Higher = more shrinkage toward league
            average.  With ~11 home matches per WSL team, 5 gives moderate
            shrinkage (~70% weight on team data for a full season).
        xg_pseudocount: Pseudo-count added to np_xG before log transform to
            avoid log(0).  Shrinks toward league prior rather than imposing a
            hard floor.
        bootstrap_n: Number of bootstrap resamples for confidence intervals.
            Set to 0 to disable.
        rho_grid: (min, max, step) for rho grid search when fitting from data.
    """

    alpha: float = 0.15
    decay_half_life_days: float = 60.0
    rho: float | None = -0.13
    max_goals: int = 8
    league_avg_goals: float = 2.6
    home_advantage_prior: float = 0.25
    penalty_shrinkage_n: float = 5.0
    xg_pseudocount: float = 0.05
    bootstrap_n: int = 0
    rho_grid: tuple[float, float, float] = (-0.30, 0.01, 0.01)


# =============================================================================
# Data Loading and Validation
# =============================================================================

REQUIRED_COLS = {
    "match_date",
    "round_label",
    "home_team",
    "away_team",
    "home_xg",
    "away_xg",
    "home_np_xg",
    "away_np_xg",
    "home_goals",
    "away_goals",
}


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    """Load CSV and validate required columns.

    round_label is preserved as a string for display/reference; it is NOT
    used for splitting (splits are date-based).  Supports labels like "R20",
    "R14", "1/4", or anything else that may come from Supabase.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError: If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Handle both ISO (YYYY-MM-DD, from Supabase) and UK (DD/MM/YYYY, from Excel)
    df["match_date"] = pd.to_datetime(df["match_date"], format="ISO8601", errors="raise")

    # Preserve round_label as string for display/reference only
    df["round_label"] = df["round_label"].astype(str).where(
        df["round_label"].notna(), None
    )

    for col in [
        "home_xg", "away_xg", "home_np_xg", "away_np_xg",
        "home_goals", "away_goals",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def split_played_future(
    df: pd.DataFrame,
    train_before: pd.Timestamp,
    predict_from: pd.Timestamp,
    predict_to: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into played (training) and future (prediction) subsets by date.

    Training set: all matches before ``train_before`` with valid xG.
    Prediction set: all matches in ``[predict_from, predict_to]`` (inclusive).

    This is entirely date-driven — round_label is not consulted, so postponed
    fixtures naturally fall into the correct bucket based on when they were
    actually played.

    Raises:
        ValueError: If training or prediction sets come out empty.
    """
    # Training set: played matches (valid xG) strictly before cutoff
    played_mask = (
        (df["match_date"] < train_before)
        & df["home_np_xg"].notna()
        & df["away_np_xg"].notna()
        & ((df["home_np_xg"] > 0) | (df["away_np_xg"] > 0))
    )
    played = df.loc[played_mask].copy()
    if played.empty:
        raise ValueError(
            f"No played matches with valid xG before {train_before.date()}"
        )

    # Prediction set: any fixtures in the date window (played or unplayed)
    future_mask = (
        (df["match_date"] >= predict_from)
        & (df["match_date"] <= predict_to)
    )
    future = df.loc[future_mask].copy()
    if future.empty:
        raise ValueError(
            f"No fixtures found in window "
            f"{predict_from.date()} to {predict_to.date()}"
        )

    logger.info(
        "Split: %d played (before %s), %d future (%s to %s)",
        len(played), train_before.date(),
        len(future), predict_from.date(), predict_to.date(),
    )
    return played, future


# =============================================================================
# Team Strength Estimation
# =============================================================================


def compute_time_weights(
    dates: pd.Series,
    reference_date: pd.Timestamp,
    half_life_days: float,
) -> np.ndarray:
    """Exponential time-decay weights.  Most recent match gets weight ≈ 1.0."""
    days_ago = (reference_date - dates).dt.days.values.astype(float)
    decay_rate = math.log(2) / half_life_days
    return np.exp(-decay_rate * days_ago)


@dataclass
class TeamStrengths:
    """Estimated team-level parameters on the log-goals scale.

    Convention:
        - Higher *attack* = stronger attacking team.
        - Higher *defence* = weaker defending team (concedes more).
        - Parameters are centred (sum-to-zero) for interpretability.
    """

    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    intercept: float

    def expected_goals(
        self, home_team: str, away_team: str
    ) -> tuple[float, float]:
        """Compute np_xG-based expected goals for a fixture.

        Returns:
            (lambda_home, lambda_away) — expected open-play goals.
        """
        log_lam_home = (
            self.intercept
            + self.home_advantage
            + self.attack.get(home_team, 0.0)
            + self.defence.get(away_team, 0.0)
        )
        log_lam_away = (
            self.intercept
            + self.attack.get(away_team, 0.0)
            + self.defence.get(home_team, 0.0)
        )
        return math.exp(log_lam_home), math.exp(log_lam_away)


def estimate_team_strengths(
    played: pd.DataFrame,
    config: ModelConfig,
) -> TeamStrengths:
    """Estimate team attack/defence strengths via weighted ridge regression.

    The model is fitted on log(np_xG) — this implies log-normal observation
    noise rather than the Poisson likelihood of canonical Dixon-Coles.  The
    trade-off is intentional: np_xG is a continuous, less noisy signal than
    discrete goal counts, which matters in small-sample leagues.

    Post-hoc sum-to-zero centering is applied for interpretability.  Because
    ridge regularisation operates on the un-centred parameterisation, this
    means the effective shrinkage target is the grand mean rather than zero.
    The practical impact is negligible with moderate α.

    Raises:
        np.linalg.LinAlgError: If the normal equations are singular (e.g.
            fewer matches than teams).  Wrapped with a descriptive message.
    """
    played = played.copy()

    teams = sorted(set(played["home_team"]) | set(played["away_team"]))
    n_teams = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}

    # Time-decay weights
    ref_date = played["match_date"].max()
    weights = compute_time_weights(
        played["match_date"], ref_date, config.decay_half_life_days
    )

    n_matches = len(played)

    # Pseudo-count floor: shrinks toward prior rather than hard clamping
    eps = config.xg_pseudocount
    home_xg = played["home_np_xg"].values + eps
    away_xg = played["away_np_xg"].values + eps

    y = np.concatenate([np.log(home_xg), np.log(away_xg)])
    w = np.concatenate([weights, weights])

    # ---- Vectorised design matrix construction ----
    home_indices = played["home_team"].map(team_to_idx).values
    away_indices = played["away_team"].map(team_to_idx).values

    n_params = 2 + 2 * n_teams
    X = np.zeros((2 * n_matches, n_params))

    rows_h = np.arange(n_matches)
    rows_a = np.arange(n_matches) + n_matches

    # Home observations: intercept + home_adv + attack[home] + defence[away]
    X[rows_h, 0] = 1.0
    X[rows_h, 1] = 1.0
    X[rows_h, 2 + home_indices] = 1.0
    X[rows_h, 2 + n_teams + away_indices] = 1.0

    # Away observations: intercept + attack[away] + defence[home]
    X[rows_a, 0] = 1.0
    X[rows_a, 2 + away_indices] = 1.0
    X[rows_a, 2 + n_teams + home_indices] = 1.0

    # ---- Weighted ridge regression: (X'WX + αI)β = X'Wy ----
    # Efficient: avoid constructing full diagonal matrix
    XtWX = (X.T * w) @ X
    XtWy = (X.T * w) @ y

    reg = config.alpha * np.eye(n_params)
    reg[0, 0] = 0.0  # Don't regularise intercept
    reg[1, 1] = config.alpha * 0.5  # Light regularisation on home advantage

    try:
        beta = np.linalg.solve(XtWX + reg, XtWy)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"Normal equations are singular — likely too few matches "
            f"({n_matches}) for {n_teams} teams.  Consider increasing "
            f"regularisation (alpha={config.alpha})."
        ) from exc

    # Extract parameters
    intercept = beta[0]
    home_advantage = beta[1]
    attack = {teams[i]: beta[2 + i] for i in range(n_teams)}
    defence = {teams[i]: beta[2 + n_teams + i] for i in range(n_teams)}

    # Sum-to-zero centering (post-hoc; see docstring)
    atk_mean = np.mean(list(attack.values()))
    def_mean = np.mean(list(defence.values()))
    attack = {t: v - atk_mean for t, v in attack.items()}
    defence = {t: v - def_mean for t, v in defence.items()}
    intercept += atk_mean + def_mean

    logger.debug(
        "Estimated strengths: intercept=%.3f, home_adv=%.3f, %d teams",
        intercept, home_advantage, n_teams,
    )

    return TeamStrengths(
        attack=attack,
        defence=defence,
        home_advantage=home_advantage,
        intercept=intercept,
    )


# =============================================================================
# Penalty Component
# =============================================================================


def estimate_penalty_rates(
    played: pd.DataFrame,
    config: ModelConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate per-team penalty xG rates with James-Stein shrinkage.

    pen_xG = xG − np_xG for each match.  Team rates are shrunk toward the
    league average using weight ``n / (n + k)`` where *k* is
    ``config.penalty_shrinkage_n``.  With ~11 home matches per WSL team per
    season and k=5, a full-season team gets ~70% weight on its own data.

    Returns:
        (home_pen_rates, away_pen_rates) — average pen xG per match.
    """
    played = played.copy()
    played["home_pen_xg"] = (played["home_xg"] - played["home_np_xg"]).clip(lower=0)
    played["away_pen_xg"] = (played["away_xg"] - played["away_np_xg"]).clip(lower=0)

    league_home_pen = played["home_pen_xg"].mean()
    league_away_pen = played["away_pen_xg"].mean()

    teams = sorted(set(played["home_team"]) | set(played["away_team"]))
    k = config.penalty_shrinkage_n

    home_pen_rates: dict[str, float] = {}
    away_pen_rates: dict[str, float] = {}

    for team in teams:
        hm = played.loc[played["home_team"] == team, "home_pen_xg"]
        if len(hm) > 0:
            sw = len(hm) / (len(hm) + k)
            home_pen_rates[team] = sw * hm.mean() + (1 - sw) * league_home_pen
        else:
            home_pen_rates[team] = league_home_pen

        am = played.loc[played["away_team"] == team, "away_pen_xg"]
        if len(am) > 0:
            sw = len(am) / (len(am) + k)
            away_pen_rates[team] = sw * am.mean() + (1 - sw) * league_away_pen
        else:
            away_pen_rates[team] = league_away_pen

    return home_pen_rates, away_pen_rates


# =============================================================================
# Scoreline Probabilities (Dixon-Coles + penalty convolution)
# =============================================================================


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function (numerically stable)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def dixon_coles_adjustment(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    """Dixon-Coles multiplicative correction for low-scoring outcomes.

    Adjusts P(0-0), P(1-0), P(0-1), P(1-1) to capture the empirical
    negative correlation between home and away goals in football.
    Returns 1.0 (no adjustment) for all other scorelines.
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    elif home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    elif home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    elif home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def _build_np_matrix(
    lambda_home_np: float,
    lambda_away_np: float,
    rho: float,
    max_goals: int,
) -> np.ndarray:
    """DC-corrected scoreline matrix from open-play (np_xG) lambdas only."""
    g = max_goals + 1
    matrix = np.zeros((g, g))
    dc_warning_emitted = False

    for h in range(g):
        for a in range(g):
            p_ind = poisson_pmf(h, lambda_home_np) * poisson_pmf(a, lambda_away_np)
            dc_adj = dixon_coles_adjustment(
                h, a, lambda_home_np, lambda_away_np, rho
            )
            if dc_adj < 0 and not dc_warning_emitted:
                logger.warning(
                    "DC adjustment went negative (%.3f) for λh=%.2f, λa=%.2f, "
                    "ρ=%.3f — clamping to 0. Consider a less extreme ρ.",
                    dc_adj, lambda_home_np, lambda_away_np, rho,
                )
                dc_warning_emitted = True
            matrix[h, a] = p_ind * max(dc_adj, 0.0)

    return matrix


def compute_scoreline_matrix(
    lambda_home_np: float,
    lambda_away_np: float,
    pen_home: float = 0.0,
    pen_away: float = 0.0,
    rho: float = -0.13,
    max_goals: int = 8,
) -> np.ndarray:
    """Joint scoreline probability matrix.

    The Dixon-Coles correction is applied to the open-play (np_xG) lambdas
    only.  Penalty goals are modelled as independent Poisson draws and
    convolved onto the open-play distribution.  This ensures the DC
    adjustment captures tactical open-play correlation without being
    distorted by penalty randomness.

    Returns:
        (max_goals+1) × (max_goals+1) matrix where entry [i, j] = P(home=i, away=j).
    """
    g = max_goals + 1
    np_matrix = _build_np_matrix(lambda_home_np, lambda_away_np, rho, max_goals)

    # If no penalty component, skip convolution
    if pen_home < 1e-6 and pen_away < 1e-6:
        total = np_matrix.sum()
        if total > 0:
            np_matrix /= total
        return np_matrix

    # Penalty PMFs
    pen_h_pmf = np.array([poisson_pmf(k, pen_home) for k in range(g)])
    pen_a_pmf = np.array([poisson_pmf(k, pen_away) for k in range(g)])

    # 2D convolution: final[h, a] = Σ np_matrix[h_np, a_np] *
    #                                  pen_h_pmf[h - h_np] * pen_a_pmf[a - a_np]
    final = np.zeros((g, g))
    for h in range(g):
        for a in range(g):
            for h_np in range(h + 1):
                h_pen = h - h_np
                for a_np in range(a + 1):
                    a_pen = a - a_np
                    final[h, a] += (
                        np_matrix[h_np, a_np]
                        * pen_h_pmf[h_pen]
                        * pen_a_pmf[a_pen]
                    )

    total = final.sum()
    if total > 0:
        final /= total

    return final


def wdl_from_matrix(
    matrix: np.ndarray,
) -> tuple[float, float, float]:
    """Extract W/D/L probabilities from a scoreline matrix.

    Returns:
        (p_home_win, p_draw, p_away_win).

    The diagonal is draws, below-diagonal entries are home wins (home scored
    more than away), and above-diagonal entries are away wins.
    """
    p_draw = float(np.trace(matrix))
    p_home_win = float(np.tril(matrix, -1).sum())
    p_away_win = float(np.triu(matrix, 1).sum())
    return p_home_win, p_draw, p_away_win


def top_scorelines(
    matrix: np.ndarray,
    k: int = 3,
) -> list[tuple[int, int, float]]:
    """Most probable *k* scorelines (heap-based, efficient for large grids)."""
    g = matrix.shape[0]

    # Use a min-heap of size k for O(g² log k) selection
    heap: list[tuple[float, int, int]] = []
    for h in range(g):
        for a in range(g):
            item = (matrix[h, a], h, a)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif matrix[h, a] > heap[0][0]:
                heapq.heapreplace(heap, item)

    # Sort descending by probability
    result = [(h, a, p) for p, h, a in sorted(heap, reverse=True)]
    return result


# =============================================================================
# Rho Fitting
# =============================================================================


def fit_rho(
    played: pd.DataFrame,
    strengths: TeamStrengths,
    config: ModelConfig,
) -> float:
    """Fit ρ by maximising the profile log-likelihood of observed goals.

    Given fixed team strength estimates (from the ridge regression on xG),
    search over a grid of ρ values and pick the one that maximises the
    log-likelihood of the *actual scorelines* under the Dixon-Coles model.

    This two-stage approach is consistent: strengths capture quality from xG,
    while ρ captures the residual correlation structure in actual goals.

    Returns:
        Optimal ρ value.
    """
    rho_min, rho_max, rho_step = config.rho_grid
    rho_values = np.arange(rho_min, rho_max + 1e-9, rho_step)

    best_rho = -0.13  # fallback
    best_ll = -np.inf

    # Pre-compute lambdas and observed goals for all matches
    home_teams = played["home_team"].values
    away_teams = played["away_team"].values
    home_goals = played["home_goals"].values.astype(int)
    away_goals = played["away_goals"].values.astype(int)

    lambdas = np.array(
        [strengths.expected_goals(h, a) for h, a in zip(home_teams, away_teams)]
    )
    lam_h = lambdas[:, 0]
    lam_a = lambdas[:, 1]

    # Pre-compute independent Poisson log-probs (ρ-independent)
    base_log_p = np.array([
        math.log(max(poisson_pmf(hg, lh) * poisson_pmf(ag, la), 1e-30))
        for hg, ag, lh, la in zip(home_goals, away_goals, lam_h, lam_a)
    ])

    for rho in rho_values:
        dc_log_adj = np.array([
            math.log(
                max(dixon_coles_adjustment(hg, ag, lh, la, rho), 1e-30)
            )
            for hg, ag, lh, la in zip(home_goals, away_goals, lam_h, lam_a)
        ])
        ll = float(np.sum(base_log_p + dc_log_adj))

        if ll > best_ll:
            best_ll = ll
            best_rho = float(rho)

    logger.info(
        "Fitted ρ = %.3f (log-likelihood = %.2f, grid [%.2f, %.2f])",
        best_rho, best_ll, rho_min, rho_max,
    )
    return best_rho


# =============================================================================
# Prediction Pipeline
# =============================================================================


@dataclass
class FixturePrediction:
    """Prediction output for a single fixture."""

    home_team: str
    away_team: str
    match_date: str
    round_label: str  # preserved from source data for display only

    # Expected goals (total = np + pen)
    lambda_home: float
    lambda_away: float
    lambda_home_np: float
    lambda_away_np: float

    # W/D/L probabilities
    p_home_win: float
    p_draw: float
    p_away_win: float

    # Top scorelines
    top_scores: list[tuple[int, int, float]]

    # Bootstrap CIs (populated if bootstrap_n > 0)
    ci_home_win: tuple[float, float] = (0.0, 0.0)
    ci_draw: tuple[float, float] = (0.0, 0.0)
    ci_away_win: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        d: dict = {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "match_date": self.match_date,
            "round": self.round_label,
            "xG_home": round(self.lambda_home, 2),
            "xG_away": round(self.lambda_away, 2),
            "np_xG_home": round(self.lambda_home_np, 2),
            "np_xG_away": round(self.lambda_away_np, 2),
            "p_home_win": round(self.p_home_win * 100, 1),
            "p_draw": round(self.p_draw * 100, 1),
            "p_away_win": round(self.p_away_win * 100, 1),
        }
        # Dynamic top-scoreline formatting (guards against < k results)
        for i, (h, a, p) in enumerate(self.top_scores, start=1):
            d[f"top{i}"] = f"{h}-{a} ({p * 100:.0f}%)"

        # Include bootstrap CIs if present
        if self.ci_home_win != (0.0, 0.0):
            d["ci_home_win"] = f"[{self.ci_home_win[0]*100:.0f}%, {self.ci_home_win[1]*100:.0f}%]"
            d["ci_draw"] = f"[{self.ci_draw[0]*100:.0f}%, {self.ci_draw[1]*100:.0f}%]"
            d["ci_away_win"] = f"[{self.ci_away_win[0]*100:.0f}%, {self.ci_away_win[1]*100:.0f}%]"

        return d


def predict_fixtures(
    future: pd.DataFrame,
    strengths: TeamStrengths,
    home_pen_rates: dict[str, float],
    away_pen_rates: dict[str, float],
    config: ModelConfig,
    rho: float = -0.13,
) -> list[FixturePrediction]:
    """Generate predictions for future fixtures."""
    predictions: list[FixturePrediction] = []

    for _, row in future.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        lam_h_np, lam_a_np = strengths.expected_goals(home_team, away_team)

        pen_h = home_pen_rates.get(home_team, 0.05)
        pen_a = away_pen_rates.get(away_team, 0.05)

        lam_h = lam_h_np + pen_h
        lam_a = lam_a_np + pen_a

        # DC applied to np_xG lambdas; penalties convolved separately
        matrix = compute_scoreline_matrix(
            lam_h_np, lam_a_np,
            pen_home=pen_h, pen_away=pen_a,
            rho=rho,
            max_goals=config.max_goals,
        )

        p_home, p_draw, p_away = wdl_from_matrix(matrix)
        tops = top_scorelines(matrix, k=3)

        predictions.append(
            FixturePrediction(
                home_team=home_team,
                away_team=away_team,
                match_date=(
                    str(row["match_date"].date())
                    if pd.notna(row["match_date"])
                    else ""
                ),
                round_label=(
                    str(row["round_label"]) if pd.notna(row["round_label"]) else ""
                ),
                lambda_home=lam_h,
                lambda_away=lam_a,
                lambda_home_np=lam_h_np,
                lambda_away_np=lam_a_np,
                p_home_win=p_home,
                p_draw=p_draw,
                p_away_win=p_away,
                top_scores=tops,
            )
        )

    return predictions


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================


def bootstrap_predictions(
    played: pd.DataFrame,
    future: pd.DataFrame,
    config: ModelConfig,
    rho: float,
    n_resamples: int = 200,
    ci_level: float = 0.90,
) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    """Compute bootstrap CIs on W/D/L probabilities.

    Resamples training matches with replacement, re-estimates team strengths
    and penalty rates, and collects prediction distributions.

    Returns:
        {(home_team, away_team): {"home_win": (lo, hi), "draw": ..., "away_win": ...}}
    """
    alpha_lo = (1 - ci_level) / 2
    alpha_hi = 1 - alpha_lo

    # Collect bootstrap samples of (p_home, p_draw, p_away) per fixture
    fixture_keys = [
        (row["home_team"], row["away_team"]) for _, row in future.iterrows()
    ]
    samples: dict[tuple[str, str], list[tuple[float, float, float]]] = {
        k: [] for k in fixture_keys
    }

    for b in range(n_resamples):
        # Resample training matches (stratified by match, not observation)
        boot = played.sample(n=len(played), replace=True)

        try:
            s = estimate_team_strengths(boot, config)
            hpr, apr = estimate_penalty_rates(boot, config)
        except (np.linalg.LinAlgError, ValueError):
            logger.debug("Bootstrap sample %d failed estimation, skipping", b)
            continue

        for _, row in future.iterrows():
            ht, at = row["home_team"], row["away_team"]
            lh_np, la_np = s.expected_goals(ht, at)
            ph = hpr.get(ht, 0.05)
            pa = apr.get(at, 0.05)

            matrix = compute_scoreline_matrix(
                lh_np, la_np,
                pen_home=ph, pen_away=pa,
                rho=rho,
                max_goals=config.max_goals,
            )
            p_h, p_d, p_a = wdl_from_matrix(matrix)
            samples[(ht, at)].append((p_h, p_d, p_a))

    # Compute percentile intervals
    cis: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for key, samps in samples.items():
        if len(samps) < 10:
            logger.warning("Fixture %s: only %d bootstrap samples", key, len(samps))
            cis[key] = {
                "home_win": (0.0, 1.0),
                "draw": (0.0, 1.0),
                "away_win": (0.0, 1.0),
            }
            continue

        arr = np.array(samps)
        cis[key] = {
            "home_win": (
                float(np.percentile(arr[:, 0], alpha_lo * 100)),
                float(np.percentile(arr[:, 0], alpha_hi * 100)),
            ),
            "draw": (
                float(np.percentile(arr[:, 1], alpha_lo * 100)),
                float(np.percentile(arr[:, 1], alpha_hi * 100)),
            ),
            "away_win": (
                float(np.percentile(arr[:, 2], alpha_lo * 100)),
                float(np.percentile(arr[:, 2], alpha_hi * 100)),
            ),
        }

    logger.info("Bootstrap: %d resamples, %.0f%% CIs", n_resamples, ci_level * 100)
    return cis


# =============================================================================
# Backtesting (date-based, weekly batches)
# =============================================================================


@dataclass
class BacktestResult:
    """Aggregated backtesting metrics."""

    n_matches: int = 0
    brier_score: float = 0.0
    log_loss: float = 0.0
    calibration_bins: list[dict] = field(default_factory=list)
    per_match: list[dict] = field(default_factory=list)


def _batch_dates_by_week(
    played_dates: pd.Series,
    start_date: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group match dates into weekly batches starting Monday.

    Returns list of (batch_start, batch_end) tuples covering all match dates
    from ``start_date`` onward.  Each tuple represents one prediction batch
    for backtesting.
    """
    dates = pd.Series(pd.to_datetime(played_dates.unique())).sort_values()
    dates = dates[dates >= start_date]
    if dates.empty:
        return []

    # Assign each date to its ISO week (year, week) tuple, then group
    batches: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_week_key: tuple[int, int] | None = None
    current_batch_dates: list[pd.Timestamp] = []

    for d in dates:
        week_key = (d.isocalendar().year, d.isocalendar().week)
        if current_week_key is None or week_key != current_week_key:
            if current_batch_dates:
                batches.append((min(current_batch_dates), max(current_batch_dates)))
            current_week_key = week_key
            current_batch_dates = [d]
        else:
            current_batch_dates.append(d)

    if current_batch_dates:
        batches.append((min(current_batch_dates), max(current_batch_dates)))

    return batches


def run_backtest(
    df: pd.DataFrame,
    config: ModelConfig,
    start_date: pd.Timestamp,
    fit_rho_each_batch: bool = False,
    min_training_matches: int = 10,
) -> BacktestResult:
    """Walk-forward backtest: train on matches before each batch, predict the batch.

    Batches are formed by grouping played match dates by ISO week.  For each
    batch, the model is trained on all matches played strictly before the
    batch's earliest date, then evaluated on the batch's matches.

    Args:
        df: Full dataset.
        config: Model configuration.
        start_date: First date to evaluate (skip earlier rounds — insufficient
            training data).
        fit_rho_each_batch: If True, re-fit ρ at each step (slower but
            simulates true out-of-sample performance).
        min_training_matches: Minimum training set size to attempt a batch.

    Returns:
        BacktestResult with aggregate and per-match metrics.
    """
    played_all = df[
        df["home_np_xg"].notna()
        & df["away_np_xg"].notna()
        & df["home_goals"].notna()
        & df["away_goals"].notna()
    ].copy()

    if played_all.empty:
        raise ValueError("No matches with valid xG and goals to backtest.")

    batches = _batch_dates_by_week(played_all["match_date"], start_date)
    if not batches:
        raise ValueError(
            f"No matches to backtest on or after {start_date.date()}"
        )

    logger.info("Backtest: %d weekly batches from %s", len(batches), start_date.date())

    per_match: list[dict] = []
    predicted_probs: list[np.ndarray] = []
    actual_outcomes: list[np.ndarray] = []

    for batch_start, batch_end in batches:
        train = played_all[played_all["match_date"] < batch_start].copy()
        test = played_all[
            (played_all["match_date"] >= batch_start)
            & (played_all["match_date"] <= batch_end)
        ].copy()

        if len(train) < min_training_matches:
            logger.debug(
                "Batch %s-%s: only %d training matches, skipping",
                batch_start.date(), batch_end.date(), len(train),
            )
            continue
        if test.empty:
            continue

        try:
            strengths = estimate_team_strengths(train, config)
            hpr, apr = estimate_penalty_rates(train, config)
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.debug(
                "Batch %s-%s: estimation failed (%s), skipping",
                batch_start.date(), batch_end.date(), exc,
            )
            continue

        # Determine ρ
        if fit_rho_each_batch:
            rho = fit_rho(train, strengths, config)
        else:
            rho = config.rho if config.rho is not None else -0.13

        preds = predict_fixtures(test, strengths, hpr, apr, config, rho=rho)

        for pred in preds:
            # Find actual result
            mask = (
                (test["home_team"] == pred.home_team)
                & (test["away_team"] == pred.away_team)
            )
            actual = test.loc[mask]
            if actual.empty or actual["home_goals"].isna().iloc[0]:
                continue

            hg = int(actual["home_goals"].iloc[0])
            ag = int(actual["away_goals"].iloc[0])

            if hg > ag:
                outcome = np.array([1.0, 0.0, 0.0])  # home win
            elif hg == ag:
                outcome = np.array([0.0, 1.0, 0.0])  # draw
            else:
                outcome = np.array([0.0, 0.0, 1.0])  # away win

            probs = np.array([pred.p_home_win, pred.p_draw, pred.p_away_win])

            predicted_probs.append(probs)
            actual_outcomes.append(outcome)
            per_match.append({
                "match_date": pred.match_date,
                "round": pred.round_label,
                "home_team": pred.home_team,
                "away_team": pred.away_team,
                "actual": f"{hg}-{ag}",
                "p_home": round(pred.p_home_win, 3),
                "p_draw": round(pred.p_draw, 3),
                "p_away": round(pred.p_away_win, 3),
                "outcome": "H" if hg > ag else ("D" if hg == ag else "A"),
            })

    if not predicted_probs:
        logger.warning("Backtest: no evaluable matches")
        return BacktestResult()

    preds_arr = np.array(predicted_probs)
    outcomes_arr = np.array(actual_outcomes)
    n = len(preds_arr)

    # Brier score (multiclass): mean squared error across outcome categories
    brier = float(np.mean(np.sum((preds_arr - outcomes_arr) ** 2, axis=1)))

    # Log-loss (multiclass)
    eps = 1e-15
    clipped = np.clip(preds_arr, eps, 1 - eps)
    log_loss_val = float(-np.mean(np.sum(outcomes_arr * np.log(clipped), axis=1)))

    # Calibration bins (using home_win probability as example)
    cal_bins = _compute_calibration_bins(
        preds_arr[:, 0], outcomes_arr[:, 0], n_bins=5
    )

    logger.info(
        "Backtest: %d matches, Brier=%.4f, LogLoss=%.4f", n, brier, log_loss_val
    )

    return BacktestResult(
        n_matches=n,
        brier_score=brier,
        log_loss=log_loss_val,
        calibration_bins=cal_bins,
        per_match=per_match,
    )


def _compute_calibration_bins(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 5,
) -> list[dict]:
    """Bin predicted probabilities and compare to observed frequencies."""
    bins = np.linspace(0, 1, n_bins + 1)
    result = []

    for i in range(n_bins):
        mask = (predicted >= bins[i]) & (predicted < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (predicted == bins[i + 1])

        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue

        result.append({
            "bin": f"{bins[i]:.0%}-{bins[i+1]:.0%}",
            "n": n_in_bin,
            "mean_predicted": round(float(predicted[mask].mean()), 3),
            "mean_observed": round(float(actual[mask].mean()), 3),
        })

    return result


# =============================================================================
# Team Summary
# =============================================================================


def team_strength_summary(
    strengths: TeamStrengths,
    home_pen_rates: dict[str, float],
    away_pen_rates: dict[str, float],
) -> pd.DataFrame:
    """Summary table of team strengths, ranked."""
    teams = sorted(strengths.attack.keys())
    rows = [
        {
            "team": team,
            "attack": round(strengths.attack[team], 3),
            "defence": round(strengths.defence[team], 3),
            "home_pen_xg": round(home_pen_rates.get(team, 0), 3),
            "away_pen_xg": round(away_pen_rates.get(team, 0), 3),
        }
        for team in teams
    ]

    df = pd.DataFrame(rows)
    df["attack_rank"] = df["attack"].rank(ascending=False).astype(int)
    df["defence_rank"] = df["defence"].rank(ascending=True).astype(int)

    return df.sort_values("attack", ascending=False).reset_index(drop=True)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WSL xG-driven match prediction model (v3, date-based)"
    )
    p.add_argument("--csv", type=str, required=True, help="Path to match data CSV")

    # Date-based splitting (replaces --as-of-round / --predict-round)
    p.add_argument(
        "--train-before", type=str, required=True,
        help="Train on matches strictly before this date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--predict-from", type=str, required=True,
        help="Earliest match_date to predict (YYYY-MM-DD, inclusive)",
    )
    p.add_argument(
        "--predict-to", type=str, required=True,
        help="Latest match_date to predict (YYYY-MM-DD, inclusive)",
    )

    p.add_argument("--alpha", type=float, default=0.15, help="Regularisation strength")
    p.add_argument(
        "--decay-days", type=float, default=60.0,
        help="Time decay half-life in days",
    )
    p.add_argument(
        "--rho", type=float, default=-0.13,
        help="Dixon-Coles ρ (ignored if --fit-rho is set)",
    )
    p.add_argument(
        "--fit-rho", action="store_true",
        help="Fit ρ from data via profile log-likelihood grid search",
    )
    p.add_argument(
        "--bootstrap", type=int, default=0,
        help="Number of bootstrap resamples for CIs (0 = disabled)",
    )
    p.add_argument(
        "--backtest", action="store_true",
        help="Run walk-forward backtest after prediction",
    )
    p.add_argument(
        "--backtest-start", type=str, default="2025-10-01",
        help="First date to evaluate in backtest (YYYY-MM-DD)",
    )
    p.add_argument("--out-predictions", type=str, default=None)
    p.add_argument("--out-strengths", type=str, default=None)
    p.add_argument("--out-backtest", type=str, default=None)
    p.add_argument(
        "-v", "--verbose", action="count", default=1,
        help="Increase verbosity (-v = INFO, -vv = DEBUG)",
    )
    p.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress all output except errors",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    verbosity = 0 if args.quiet else args.verbose
    configure_logging(verbosity)

    config = ModelConfig(
        alpha=args.alpha,
        decay_half_life_days=args.decay_days,
        rho=None if args.fit_rho else args.rho,
        bootstrap_n=args.bootstrap,
    )

    # Parse dates
    train_before = pd.Timestamp(args.train_before)
    predict_from = pd.Timestamp(args.predict_from)
    predict_to = pd.Timestamp(args.predict_to)

    # Load data
    df = load_and_validate(Path(args.csv))

    # Split by date
    played, future = split_played_future(
        df, train_before, predict_from, predict_to
    )

    logger.info(
        "Training: %d matches before %s | Predicting: %d fixtures (%s to %s)",
        len(played), train_before.date(),
        len(future), predict_from.date(), predict_to.date(),
    )
    logger.info(
        "Config: α=%.3f, decay=%.0fd, ρ=%s, bootstrap=%d",
        config.alpha, config.decay_half_life_days,
        "fit" if config.rho is None else f"{config.rho:.3f}",
        config.bootstrap_n,
    )

    # Estimate team strengths
    strengths = estimate_team_strengths(played, config)
    logger.info(
        "Home advantage: %.3f (%.2f× goals multiplier)",
        strengths.home_advantage, math.exp(strengths.home_advantage),
    )
    logger.info(
        "Intercept: %.3f (%.2f base goals per team)",
        strengths.intercept, math.exp(strengths.intercept),
    )

    # Fit or use fixed ρ
    if config.rho is None:
        rho = fit_rho(played, strengths, config)
    else:
        rho = config.rho

    # Estimate penalty rates
    home_pen_rates, away_pen_rates = estimate_penalty_rates(played, config)

    # Generate predictions
    predictions = predict_fixtures(
        future, strengths, home_pen_rates, away_pen_rates, config, rho=rho
    )

    # Bootstrap CIs
    if config.bootstrap_n > 0:
        cis = bootstrap_predictions(
            played, future, config, rho=rho, n_resamples=config.bootstrap_n
        )
        for pred in predictions:
            key = (pred.home_team, pred.away_team)
            if key in cis:
                pred.ci_home_win = cis[key]["home_win"]
                pred.ci_draw = cis[key]["draw"]
                pred.ci_away_win = cis[key]["away_win"]

    # --- Display predictions ---
    print("=" * 80)
    print(
        f"PREDICTIONS: {predict_from.date()} to {predict_to.date()} "
        f"({len(predictions)} fixtures)"
    )
    print("=" * 80)

    pred_df = pd.DataFrame([p.to_dict() for p in predictions])
    print(pred_df.to_string(index=False))
    print()

    # --- Display team strengths ---
    print("=" * 80)
    print("TEAM STRENGTHS (attack: higher=better, defence: lower=better)")
    print("=" * 80)

    strength_df = team_strength_summary(strengths, home_pen_rates, away_pen_rates)
    print(strength_df.to_string(index=False))

    # --- Save outputs ---
    if args.out_predictions:
        pred_df.to_csv(args.out_predictions, index=False)
        logger.info("Saved predictions → %s", args.out_predictions)

    if args.out_strengths:
        strength_df.to_csv(args.out_strengths, index=False)
        logger.info("Saved team strengths → %s", args.out_strengths)

    # --- Backtest ---
    if args.backtest:
        print()
        print("=" * 80)
        print("WALK-FORWARD BACKTEST (weekly batches)")
        print("=" * 80)

        bt = run_backtest(
            df, config,
            start_date=pd.Timestamp(args.backtest_start),
            fit_rho_each_batch=(config.rho is None),
        )

        print(f"Matches evaluated:  {bt.n_matches}")
        print(f"Brier score:        {bt.brier_score:.4f}")
        print(f"Log-loss:           {bt.log_loss:.4f}")
        print()

        if bt.calibration_bins:
            print("Calibration (home win probability):")
            cal_df = pd.DataFrame(bt.calibration_bins)
            print(cal_df.to_string(index=False))

        if args.out_backtest:
            bt_df = pd.DataFrame(bt.per_match)
            bt_df.to_csv(args.out_backtest, index=False)
            logger.info("Saved backtest results → %s", args.out_backtest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())