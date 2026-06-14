# Simulation Strategy

Last updated: 2026-06-14

The simulation baseline answers a different question from logged replay evaluation.

Logged replay evaluation measures whether stored dashboard predictions for Matchweeks 2-22 were well calibrated against actual results. Monte Carlo simulation uses the model's expected-goals outputs to estimate a distribution of possible final league tables from a chosen point in the season.

## Mid-Season Use Case

The first simulation target is:

- Season: 2025-26
- Cutoff: after Matchweek 11
- Starting table: actual results from Matchweeks 1-11
- Simulated fixtures: Matchweeks 12-22
- Outputs: title probability, top-3 probability, top-4 probability, average final points, average final goal difference, average rank, and rank distribution

Week 1 is included in the actual starting table for this use case. The separate Week 1 historical-prior problem only applies when trying to predict Matchweek 1 before any current-season matches exist.

## Why Scoreline Simulation

The prediction model already exposes expected-goals lambdas through `FixturePrediction.lambda_home` and `FixturePrediction.lambda_away`. The simulation therefore samples home and away goals from Poisson distributions instead of sampling only home/draw/away outcomes.

Scoreline simulation is preferred because it updates:

- Goals for
- Goals against
- Goal difference
- Wins, draws, losses
- Points

That matters because WSL table ranking uses points, goal difference, goals for, and then a deterministic fallback in this baseline.

## Data Flow

```text
Supabase rpc_wsl_weekly_stats()
        ->
actual results through cutoff week
        ->
starting league table

Supabase rpc_wsl_weekly_stats()
        ->
remaining fixtures by round_label
        ->
existing xG model prediction pathway
        ->
expected-goals lambdas
        ->
Poisson scoreline simulations
        ->
final table probability summary
```

The simulation module does not call live Supabase in unit tests. Tests use mocked DataFrames and pure functions.

## Running The Simulation

```bash
python scripts/run_monte_carlo_simulation.py \
  --season 2025-26 \
  --cutoff-week 11 \
  --remaining-start-week 12 \
  --remaining-end-week 22 \
  --simulations 10000 \
  --random-seed 42 \
  --output reports/monte_carlo_after_week_11_2025_26.md
```

The script:

- Fetches match data from Supabase using the existing data layer.
- Builds the actual table from results through Matchweek 11.
- Trains the existing model on completed fixtures through Matchweek 11.
- Predicts expected-goals lambdas for Matchweeks 12-22.
- Samples scorelines with a fixed random seed for reproducibility.
- Writes an interview-ready Markdown report when `--output` is provided.

## Relationship To Logged Replay Evaluation

Use logged replay evaluation to say, "This model achieved Brier 0.519150, log loss 0.891373, and accuracy 0.611111 on 126 replayed fixtures."

Use Monte Carlo simulation to say, "Given the actual table after Matchweek 11 and the model's expected-goals outlook for the remaining fixtures, these were the estimated title, top-3, and top-4 probabilities."

Together, the two workflows show both measured prediction quality and decision-facing scenario analysis.

## Interview Discussion Value

This supports a Genius Sports-style discussion because it demonstrates:

- Backtested model quality before relying on simulation outputs.
- Scoreline-level simulation rather than coarse H/D/A-only sampling.
- Reproducible results through explicit seeds and documented inputs.
- Proper table mechanics, including goal difference.
- A clean separation between model logic, evaluation, simulation, API, and dashboard operations.

## Current Limitations

- Poisson goal draws assume conditional independence after lambdas are estimated.
- The simulation does not model injuries, suspensions, lineup rotation, transfers, tactical changes, weather, or other contextual factors.
- It uses the existing model pathway and does not add a new model family.
- It does not implement Monte Carlo-triggering from the dashboard.
- It does not persist simulation runs to Supabase yet.
