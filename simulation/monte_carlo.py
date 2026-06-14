"""
simulation/monte_carlo.py
-------------------------
Poisson scoreline-based WSL season Monte Carlo simulation.
"""

from __future__ import annotations

from collections import defaultdict
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
    if not np.isfinite(lambda_home) or not np.isfinite(lambda_away) or lambda_home < 0 or lambda_away < 0:
        raise ValueError("Expected-goals lambdas must be finite and non-negative.")
    return int(rng.poisson(lambda_home)), int(rng.poisson(lambda_away))


def simulate_remaining_fixtures(
    starting_table: pd.DataFrame,
    fixtures: list[SimFixture],
    rng: np.random.Generator,
) -> pd.DataFrame:
    table = starting_table.drop(columns=["rank"], errors="ignore").copy()
    for fixture in fixtures:
        home_goals, away_goals = sample_scoreline(fixture.lambda_home, fixture.lambda_away, rng)
        table = apply_result(table, fixture.home_team, fixture.away_team, home_goals, away_goals)
    return rank_table(table)


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
    teams = sorted(
        set(starting_table["team"])
        .union({fixture.home_team for fixture in fixtures})
        .union({fixture.away_team for fixture in fixtures})
    )
    accum = {
        team: {
            "titles": 0,
            "top3": 0,
            "top4": 0,
            "points": [],
            "gd": [],
            "rank": [],
            "rank_distribution": defaultdict(int),
        }
        for team in teams
    }

    for _ in range(simulations):
        final_table = simulate_remaining_fixtures(starting_table, fixtures, rng)
        for row in final_table.to_dict(orient="records"):
            team = row["team"]
            rank = int(row["rank"])
            accum[team]["titles"] += int(rank == 1)
            accum[team]["top3"] += int(rank <= 3)
            accum[team]["top4"] += int(rank <= 4)
            accum[team]["points"].append(float(row["points"]))
            accum[team]["gd"].append(float(row["gd"]))
            accum[team]["rank"].append(float(rank))
            accum[team]["rank_distribution"][rank] += 1

    summary = []
    for team in teams:
        team_data = accum[team]
        summary.append(
            {
                "team": team,
                "title_probability": team_data["titles"] / simulations,
                "top3_probability": team_data["top3"] / simulations,
                "top4_probability": team_data["top4"] / simulations,
                "average_final_points": float(np.mean(team_data["points"])),
                "average_final_goal_difference": float(np.mean(team_data["gd"])),
                "average_rank": float(np.mean(team_data["rank"])),
                "rank_distribution": dict(sorted(team_data["rank_distribution"].items())),
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
