from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.run_monte_carlo_simulation import build_parser
from simulation.monte_carlo import (
    SimFixture,
    apply_result,
    build_markdown_report,
    build_table_from_results,
    rank_table,
    run_monte_carlo,
    sample_scoreline,
    simulate_remaining_fixtures,
)


def test_build_table_from_actual_results_through_cutoff_week():
    results = pd.DataFrame(
        [
            {"round_label": "R1", "home_team": "Arsenal", "away_team": "Chelsea", "home_goals": 2, "away_goals": 1},
            {"round_label": "R1", "home_team": "City", "away_team": "United", "home_goals": 0, "away_goals": 0},
            {"round_label": "R2", "home_team": "Chelsea", "away_team": "City", "home_goals": 3, "away_goals": 1},
            {"round_label": "R3", "home_team": "United", "away_team": "Arsenal", "home_goals": 4, "away_goals": 1},
        ]
    )

    table = build_table_from_results(results, cutoff_week=2)

    arsenal = table.loc[table["team"] == "Arsenal"].iloc[0]
    chelsea = table.loc[table["team"] == "Chelsea"].iloc[0]
    assert int(arsenal["points"]) == 3
    assert int(chelsea["points"]) == 3
    assert int(chelsea["gf"]) == 4
    assert set(table["team"]) == {"Arsenal", "Chelsea", "City", "United"}


def test_poisson_scoreline_sampling_is_reproducible_with_seed():
    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    assert sample_scoreline(1.4, 0.8, first_rng) == sample_scoreline(1.4, 0.8, second_rng)


def test_apply_result_updates_table_with_goals_and_points():
    table = pd.DataFrame(
        [
            {"team": "Arsenal", "played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "gd": 0, "points": 0},
            {"team": "Chelsea", "played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "gd": 0, "points": 0},
        ]
    )

    updated = apply_result(table, "Arsenal", "Chelsea", 2, 1)

    arsenal = updated.loc[updated["team"] == "Arsenal"].iloc[0]
    chelsea = updated.loc[updated["team"] == "Chelsea"].iloc[0]
    assert int(arsenal["wins"]) == 1
    assert int(arsenal["points"]) == 3
    assert int(arsenal["gd"]) == 1
    assert int(chelsea["losses"]) == 1
    assert int(chelsea["ga"]) == 2


def test_ranking_uses_points_goal_difference_goals_for_then_team_name():
    table = pd.DataFrame(
        [
            {"team": "Beta", "played": 4, "wins": 3, "draws": 1, "losses": 0, "gf": 7, "ga": 2, "gd": 5, "points": 10},
            {"team": "Alpha", "played": 4, "wins": 3, "draws": 1, "losses": 0, "gf": 8, "ga": 3, "gd": 5, "points": 10},
            {"team": "Delta", "played": 4, "wins": 3, "draws": 1, "losses": 0, "gf": 8, "ga": 3, "gd": 5, "points": 10},
            {"team": "Gamma", "played": 4, "wins": 3, "draws": 0, "losses": 1, "gf": 20, "ga": 2, "gd": 18, "points": 9},
        ]
    )

    ranked = rank_table(table)

    assert ranked["team"].tolist() == ["Alpha", "Delta", "Beta", "Gamma"]


def test_simulate_remaining_fixtures_updates_table_from_sampled_scorelines():
    starting_table = rank_table(
        pd.DataFrame(
            [
                {"team": "Arsenal", "played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "gd": 0, "points": 0},
                {"team": "Chelsea", "played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "gd": 0, "points": 0},
            ]
        )
    )
    fixtures = [
        SimFixture(
            week=12,
            match_date="2026-01-10",
            home_team="Arsenal",
            away_team="Chelsea",
            lambda_home=1.8,
            lambda_away=0.6,
        )
    ]

    final_table = simulate_remaining_fixtures(starting_table, fixtures, np.random.default_rng(7))

    assert int(final_table["played"].sum()) == 2
    assert int(final_table["gf"].sum()) == int(final_table["ga"].sum())


def test_monte_carlo_probabilities_and_rank_distribution_are_reported():
    starting_table = rank_table(
        pd.DataFrame(
            [
                {"team": "Arsenal", "played": 11, "wins": 9, "draws": 1, "losses": 1, "gf": 25, "ga": 5, "gd": 20, "points": 28},
                {"team": "Chelsea", "played": 11, "wins": 8, "draws": 1, "losses": 2, "gf": 20, "ga": 8, "gd": 12, "points": 25},
                {"team": "City", "played": 11, "wins": 6, "draws": 2, "losses": 3, "gf": 17, "ga": 12, "gd": 5, "points": 20},
                {"team": "United", "played": 11, "wins": 4, "draws": 2, "losses": 5, "gf": 14, "ga": 15, "gd": -1, "points": 14},
            ]
        )
    )

    result = run_monte_carlo(starting_table, [], simulations=5, random_seed=42)
    arsenal = next(row for row in result["summary"] if row["team"] == "Arsenal")
    united = next(row for row in result["summary"] if row["team"] == "United")

    assert arsenal["title_probability"] == 1.0
    assert arsenal["top3_probability"] == 1.0
    assert united["top4_probability"] == 1.0
    assert arsenal["rank_distribution"] == {1: 5}


def test_monte_carlo_results_are_reproducible():
    starting_table = rank_table(
        pd.DataFrame(
            [
                {"team": "A", "played": 1, "wins": 1, "draws": 0, "losses": 0, "gf": 2, "ga": 0, "gd": 2, "points": 3},
                {"team": "B", "played": 1, "wins": 0, "draws": 0, "losses": 1, "gf": 0, "ga": 2, "gd": -2, "points": 0},
            ]
        )
    )
    fixtures = [SimFixture(week=12, match_date="2026-01-10", home_team="B", away_team="A", lambda_home=1.1, lambda_away=1.0)]

    first = run_monte_carlo(starting_table, fixtures, simulations=50, random_seed=99)
    second = run_monte_carlo(starting_table, fixtures, simulations=50, random_seed=99)

    assert first == second


def test_markdown_report_contains_required_sections():
    starting_table = rank_table(
        pd.DataFrame(
            [
                {"team": "Arsenal", "played": 11, "wins": 9, "draws": 1, "losses": 1, "gf": 25, "ga": 5, "gd": 20, "points": 28},
            ]
        )
    )
    result = run_monte_carlo(starting_table, [], simulations=2, random_seed=42)

    report = build_markdown_report(
        season="2025-26",
        cutoff_week=11,
        starting_table=starting_table,
        simulation_result=result,
    )

    assert "# WSL 2025-26 Monte Carlo After Matchweek 11" in report
    assert "## Title Probabilities" in report
    assert "## Rank Distribution Summary" in report
    assert "Scorelines are sampled using Poisson distributions" in report


def test_cli_argument_parsing_for_midseason_simulation_command():
    args = build_parser().parse_args(
        [
            "--season",
            "2025-26",
            "--cutoff-week",
            "11",
            "--remaining-start-week",
            "12",
            "--remaining-end-week",
            "22",
            "--simulations",
            "10000",
            "--random-seed",
            "42",
            "--output",
            "reports/monte_carlo_after_week_11_2025_26.md",
        ]
    )

    assert isinstance(args, argparse.Namespace)
    assert args.season == "2025-26"
    assert args.cutoff_week == 11
    assert args.remaining_start_week == 12
    assert args.remaining_end_week == 22
    assert args.simulations == 10000
