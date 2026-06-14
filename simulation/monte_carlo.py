"""
simulation/monte_carlo.py
-------------------------
Poisson scoreline-based WSL season Monte Carlo simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimFixture:
    week: int
    match_date: str
    home_team: str
    away_team: str
    lambda_home: float
    lambda_away: float


TABLE_COLUMNS = ["team", "played", "wins", "draws", "losses", "gf", "ga", "gd", "points"]
STATE_COLUMNS = ["played", "wins", "draws", "losses", "gf", "ga", "points"]
PLAYED, WINS, DRAWS, LOSSES, GF, GA, POINTS = range(len(STATE_COLUMNS))


def week_from_round_label(round_label: Any) -> int | None:
    digits = "".join(char for char in str(round_label) if char.isdigit())
    return int(digits) if digits else None


def empty_table(teams: list[str]) -> pd.DataFrame:
    rows = [
        {
            "team": team,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "points": 0,
        }
        for team in sorted(set(teams))
    ]
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def apply_result(
    table: pd.DataFrame,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> pd.DataFrame:
    table = _ensure_teams(table, [home_team, away_team]).copy()
    home_idx = table.index[table["team"] == home_team][0]
    away_idx = table.index[table["team"] == away_team][0]

    table.loc[home_idx, ["played", "gf", "ga"]] += [1, home_goals, away_goals]
    table.loc[away_idx, ["played", "gf", "ga"]] += [1, away_goals, home_goals]

    if home_goals > away_goals:
        table.loc[home_idx, ["wins", "points"]] += [1, 3]
        table.loc[away_idx, "losses"] += 1
    elif home_goals < away_goals:
        table.loc[away_idx, ["wins", "points"]] += [1, 3]
        table.loc[home_idx, "losses"] += 1
    else:
        table.loc[home_idx, ["draws", "points"]] += [1, 1]
        table.loc[away_idx, ["draws", "points"]] += [1, 1]

    table["gd"] = table["gf"] - table["ga"]
    return table


def _ensure_teams(table: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    existing = set(table["team"]) if not table.empty else set()
    missing = sorted(set(teams) - existing)
    if not missing:
        return table
    return pd.concat([table, empty_table(missing)], ignore_index=True)


def build_table_from_results(results_df: pd.DataFrame, cutoff_week: int) -> pd.DataFrame:
    required = {"round_label", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(f"Results data missing required columns: {sorted(missing)}")

    work = results_df.copy()
    work["week"] = work["round_label"].map(week_from_round_label)
    work = work[
        (work["week"].notna())
        & (work["week"] <= cutoff_week)
        & work["home_goals"].notna()
        & work["away_goals"].notna()
    ]

    teams = sorted(set(work["home_team"]).union(set(work["away_team"])))
    table = empty_table([str(team) for team in teams])
    for row in work.to_dict(orient="records"):
        table = apply_result(
            table,
            str(row["home_team"]),
            str(row["away_team"]),
            int(row["home_goals"]),
            int(row["away_goals"]),
        )
    return rank_table(table)


def rank_table(table: pd.DataFrame) -> pd.DataFrame:
    ranked = table.copy()
    ranked = ranked.drop(columns=["rank"], errors="ignore")
    ranked["gd"] = ranked["gf"] - ranked["ga"]
    ranked = ranked.sort_values(
        ["points", "gd", "gf", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def sample_scoreline(lambda_home: float, lambda_away: float, rng: np.random.Generator) -> tuple[int, int]:
    if (
        not np.isfinite(lambda_home)
        or not np.isfinite(lambda_away)
        or lambda_home < 0
        or lambda_away < 0
    ):
        raise ValueError("Expected-goals lambdas must be finite and non-negative.")
    return int(rng.poisson(lambda_home)), int(rng.poisson(lambda_away))


def simulate_remaining_fixtures(
    starting_table: pd.DataFrame,
    fixtures: list[SimFixture],
    rng: np.random.Generator,
) -> pd.DataFrame:
    teams = _simulation_teams(starting_table, fixtures)
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    state = _table_to_state(starting_table, teams)
    home_indices, away_indices, lambda_home, lambda_away = _fixture_arrays(fixtures, team_to_idx)
    home_goals = rng.poisson(lambda_home).astype(np.int64)
    away_goals = rng.poisson(lambda_away).astype(np.int64)
    _apply_fixture_arrays(state, home_indices, away_indices, home_goals, away_goals)
    return _state_to_ranked_table(teams, state)


def run_monte_carlo(
    starting_table: pd.DataFrame,
    fixtures: list[SimFixture],
    *,
    simulations: int,
    random_seed: int,
) -> dict[str, Any]:
    if simulations <= 0:
        raise ValueError("simulations must be positive.")

    rng = np.random.default_rng(random_seed)
    teams = _simulation_teams(starting_table, fixtures)
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    base_state = _table_to_state(starting_table, teams)
    home_indices, away_indices, lambda_home, lambda_away = _fixture_arrays(fixtures, team_to_idx)

    n_teams = len(teams)
    titles = np.zeros(n_teams, dtype=np.int64)
    top3 = np.zeros(n_teams, dtype=np.int64)
    top4 = np.zeros(n_teams, dtype=np.int64)
    points_sum = np.zeros(n_teams, dtype=np.float64)
    gd_sum = np.zeros(n_teams, dtype=np.float64)
    rank_sum = np.zeros(n_teams, dtype=np.float64)
    rank_counts = np.zeros((n_teams, n_teams), dtype=np.int64)

    for _ in range(simulations):
        state = base_state.copy()
        home_goals = rng.poisson(lambda_home).astype(np.int64)
        away_goals = rng.poisson(lambda_away).astype(np.int64)
        _apply_fixture_arrays(state, home_indices, away_indices, home_goals, away_goals)

        gd = state[:, GF] - state[:, GA]
        ranked_indices = _rank_indices(teams, state, gd)
        ranks = np.empty(n_teams, dtype=np.int64)
        ranks[ranked_indices] = np.arange(1, n_teams + 1)

        titles += ranks == 1
        top3 += ranks <= 3
        top4 += ranks <= 4
        points_sum += state[:, POINTS]
        gd_sum += gd
        rank_sum += ranks
        rank_counts[np.arange(n_teams), ranks - 1] += 1

    summary = []
    for idx, team in enumerate(teams):
        summary.append(
            {
                "team": team,
                "title_probability": float(titles[idx] / simulations),
                "top3_probability": float(top3[idx] / simulations),
                "top4_probability": float(top4[idx] / simulations),
                "average_final_points": float(points_sum[idx] / simulations),
                "average_final_goal_difference": float(gd_sum[idx] / simulations),
                "average_rank": float(rank_sum[idx] / simulations),
                "rank_distribution": {
                    rank: int(count)
                    for rank, count in enumerate(rank_counts[idx], start=1)
                    if count
                },
            }
        )

    summary = sorted(
        summary,
        key=lambda row: (-row["title_probability"], -row["top3_probability"], row["average_rank"], row["team"]),
    )
    return {
        "simulations": simulations,
        "random_seed": random_seed,
        "remaining_fixture_count": len(fixtures),
        "summary": summary,
    }


def _simulation_teams(starting_table: pd.DataFrame, fixtures: list[SimFixture]) -> list[str]:
    return sorted(
        set(starting_table["team"])
        .union({fixture.home_team for fixture in fixtures})
        .union({fixture.away_team for fixture in fixtures})
    )


def _table_to_state(table: pd.DataFrame, teams: list[str]) -> np.ndarray:
    work = table.drop(columns=["rank"], errors="ignore").copy()
    work = work.set_index("team").reindex(teams, fill_value=0)
    for column in STATE_COLUMNS:
        if column not in work.columns:
            work[column] = 0
    return work[STATE_COLUMNS].to_numpy(dtype=np.int64)


def _state_to_ranked_table(teams: list[str], state: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, team in enumerate(teams):
        gf = int(state[idx, GF])
        ga = int(state[idx, GA])
        rows.append(
            {
                "team": team,
                "played": int(state[idx, PLAYED]),
                "wins": int(state[idx, WINS]),
                "draws": int(state[idx, DRAWS]),
                "losses": int(state[idx, LOSSES]),
                "gf": gf,
                "ga": ga,
                "gd": gf - ga,
                "points": int(state[idx, POINTS]),
            }
        )
    return rank_table(pd.DataFrame(rows, columns=TABLE_COLUMNS))


def _fixture_arrays(
    fixtures: list[SimFixture],
    team_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    home_indices = np.array([team_to_idx[fixture.home_team] for fixture in fixtures], dtype=np.int64)
    away_indices = np.array([team_to_idx[fixture.away_team] for fixture in fixtures], dtype=np.int64)
    lambda_home = np.array([fixture.lambda_home for fixture in fixtures], dtype=np.float64)
    lambda_away = np.array([fixture.lambda_away for fixture in fixtures], dtype=np.float64)

    if np.any(~np.isfinite(lambda_home)) or np.any(~np.isfinite(lambda_away)):
        raise ValueError("Expected-goals lambdas must be finite and non-negative.")
    if np.any(lambda_home < 0) or np.any(lambda_away < 0):
        raise ValueError("Expected-goals lambdas must be finite and non-negative.")

    return home_indices, away_indices, lambda_home, lambda_away


def _apply_fixture_arrays(
    state: np.ndarray,
    home_indices: np.ndarray,
    away_indices: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> None:
    for idx, home_idx in enumerate(home_indices):
        away_idx = away_indices[idx]
        hg = int(home_goals[idx])
        ag = int(away_goals[idx])

        state[home_idx, PLAYED] += 1
        state[away_idx, PLAYED] += 1
        state[home_idx, GF] += hg
        state[home_idx, GA] += ag
        state[away_idx, GF] += ag
        state[away_idx, GA] += hg

        if hg > ag:
            state[home_idx, WINS] += 1
            state[home_idx, POINTS] += 3
            state[away_idx, LOSSES] += 1
        elif hg < ag:
            state[away_idx, WINS] += 1
            state[away_idx, POINTS] += 3
            state[home_idx, LOSSES] += 1
        else:
            state[home_idx, DRAWS] += 1
            state[away_idx, DRAWS] += 1
            state[home_idx, POINTS] += 1
            state[away_idx, POINTS] += 1


def _rank_indices(teams: list[str], state: np.ndarray, gd: np.ndarray) -> list[int]:
    return sorted(
        range(len(teams)),
        key=lambda idx: (-state[idx, POINTS], -gd[idx], -state[idx, GF], teams[idx]),
    )


def build_markdown_report(
    *,
    season: str,
    cutoff_week: int,
    starting_table: pd.DataFrame,
    simulation_result: dict[str, Any],
) -> str:
    lines = [
        f"# WSL {season} Monte Carlo After Matchweek {cutoff_week}",
        "",
        "## Summary",
        "",
        f"- Season: {season}",
        f"- Cutoff week: {cutoff_week}",
        f"- Remaining fixture count: {simulation_result['remaining_fixture_count']}",
        f"- Simulations: {simulation_result['simulations']}",
        f"- Random seed: {simulation_result['random_seed']}",
        "",
        "## Actual Table After Cutoff",
        "",
        *_markdown_table(starting_table.to_dict(orient="records")),
        "",
        "## Title Probabilities",
        "",
        *_markdown_table(_probability_rows(simulation_result["summary"], "title_probability")),
        "",
        "## Top-3 Probabilities",
        "",
        *_markdown_table(_probability_rows(simulation_result["summary"], "top3_probability")),
        "",
        "## Top-4 Probabilities",
        "",
        *_markdown_table(_probability_rows(simulation_result["summary"], "top4_probability")),
        "",
        "## Expected Final Table Metrics",
        "",
        *_markdown_table(_final_metric_rows(simulation_result["summary"])),
        "",
        "## Rank Distribution Summary",
        "",
        *_markdown_table(_rank_distribution_rows(simulation_result["summary"])),
        "",
        "## Notes",
        "",
        "- Week 1 is included in the actual starting table.",
        "- Remaining fixtures are simulated from model expected-goals outputs.",
        "- Scorelines are sampled using Poisson distributions so goals for, goals against and goal difference update.",
        "- The simulation does not account for squad news, injuries, tactical changes, schedule congestion or other non-model context.",
    ]
    return "\n".join(lines) + "\n"


def _probability_rows(summary: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [{"team": row["team"], key: round(row[key], 4)} for row in summary]


def _final_metric_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "team": row["team"],
            "average_final_points": round(row["average_final_points"], 2),
            "average_final_goal_difference": round(row["average_final_goal_difference"], 2),
            "average_rank": round(row["average_rank"], 2),
        }
        for row in sorted(summary, key=lambda item: item["average_rank"])
    ]


def _rank_distribution_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "team": row["team"],
            "rank_distribution": row["rank_distribution"],
        }
        for row in sorted(summary, key=lambda item: item["average_rank"])
    ]


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["_None_"]
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines
