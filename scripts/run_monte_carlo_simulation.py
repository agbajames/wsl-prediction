"""
scripts/run_monte_carlo_simulation.py
-------------------------------------
Run a Poisson scoreline-based mid-season WSL Monte Carlo simulation.

Run locally:
    python scripts/run_monte_carlo_simulation.py \
      --season 2025-26 \
      --cutoff-week 11 \
      --remaining-start-week 12 \
      --remaining-end-week 22 \
      --simulations 10000 \
      --random-seed 42 \
      --output reports/monte_carlo_after_week_11_2025_26.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.supabase_client import fetch_match_data
from model.wsl_xg_model import (
    ModelConfig,
    estimate_penalty_rates,
    estimate_team_strengths,
    fit_rho,
    predict_fixtures,
)
from simulation.monte_carlo import (
    SimFixture,
    build_markdown_report,
    build_table_from_results,
    run_monte_carlo,
    week_from_round_label,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a WSL mid-season Poisson scoreline Monte Carlo simulation."
    )
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--cutoff-week", type=int, default=11)
    parser.add_argument("--remaining-start-week", type=int, default=12)
    parser.add_argument("--remaining-end-week", type=int, default=22)
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fit-rho", action="store_true")
    return parser


def filter_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Filter a WSL season label such as 2025-26 to the football-year window."""
    start_year = int(season.split("-")[0])
    start = pd.Timestamp(f"{start_year}-07-01")
    end = pd.Timestamp(f"{start_year + 1}-07-01")
    return df[(df["match_date"] >= start) & (df["match_date"] < end)].copy()


def build_remaining_fixtures_from_model(
    df: pd.DataFrame,
    *,
    cutoff_week: int,
    remaining_start_week: int,
    remaining_end_week: int,
    fit_rho_each_run: bool = False,
) -> list[SimFixture]:
    """Train on completed fixtures through the cutoff and predict remaining lambdas."""
    work = df.copy()
    work["week"] = work["round_label"].map(week_from_round_label)

    played = work[
        (work["week"].notna())
        & (work["week"] <= cutoff_week)
        & work["home_np_xg"].notna()
        & work["away_np_xg"].notna()
        & ((work["home_np_xg"] > 0) | (work["away_np_xg"] > 0))
    ].copy()
    remaining = work[
        (work["week"].notna())
        & (work["week"] >= remaining_start_week)
        & (work["week"] <= remaining_end_week)
    ].copy()

    if played.empty:
        raise ValueError(f"No completed fixtures found through Matchweek {cutoff_week}.")
    if remaining.empty:
        raise ValueError(
            "No remaining fixtures found for "
            f"Matchweeks {remaining_start_week}-{remaining_end_week}."
        )

    config = ModelConfig(rho=None if fit_rho_each_run else -0.13)
    strengths = estimate_team_strengths(played, config)
    rho = fit_rho(played, strengths, config) if config.rho is None else config.rho
    home_pen_rates, away_pen_rates = estimate_penalty_rates(played, config)
    predictions = predict_fixtures(
        remaining,
        strengths,
        home_pen_rates,
        away_pen_rates,
        config,
        rho=rho,
    )

    return [
        SimFixture(
            week=week_from_round_label(prediction.round_label) or 0,
            match_date=prediction.match_date,
            home_team=prediction.home_team,
            away_team=prediction.away_team,
            lambda_home=prediction.lambda_home,
            lambda_away=prediction.lambda_away,
        )
        for prediction in predictions
    ]


def run_from_dataframe(args: argparse.Namespace, df: pd.DataFrame) -> dict[str, Any]:
    season_df = filter_season(df, args.season)
    starting_table = build_table_from_results(season_df, args.cutoff_week)
    remaining_fixtures = build_remaining_fixtures_from_model(
        season_df,
        cutoff_week=args.cutoff_week,
        remaining_start_week=args.remaining_start_week,
        remaining_end_week=args.remaining_end_week,
        fit_rho_each_run=args.fit_rho,
    )
    simulation_result = run_monte_carlo(
        starting_table,
        remaining_fixtures,
        simulations=args.simulations,
        random_seed=args.random_seed,
    )

    if args.output:
        report = build_markdown_report(
            season=args.season,
            cutoff_week=args.cutoff_week,
            starting_table=starting_table,
            simulation_result=simulation_result,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")

    return {
        "season": args.season,
        "cutoff_week": args.cutoff_week,
        "remaining_start_week": args.remaining_start_week,
        "remaining_end_week": args.remaining_end_week,
        "output": str(args.output) if args.output else None,
        **simulation_result,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = fetch_match_data()
    result = run_from_dataframe(args, df)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
