# Poisson Regression Challenger

## Why this model exists

`poisson_regression` is an offline challenger for the WSL model evaluation
framework. It adds another interpretable statistical benchmark without changing
the production champion model, API, dashboard, or live prediction outputs.

The model is intended to answer a narrow question: can a simple regularised
scoring-rate regression beat the existing offline challengers while remaining
easy to inspect on small WSL samples?

## How it differs from existing models

- `elo_baseline` updates team ratings from match outcomes and maps rating gaps
  to H/D/A probabilities.
- `logistic_regression` directly models match outcome classes from historical
  team-form features.
- `regularised_team_strength` estimates smoothed attack and defence ratios, then
  uses a Poisson score grid.
- `champion_dc_xg` is the production Dixon-Coles-style xG model and remains the
  model to beat.
- `poisson_regression` fits separate ridge-regularised Poisson regressions for
  home and away scoring rates, then aggregates scoreline probabilities into
  H/D/A probabilities.

## How the model works

For each backtest fold, the model trains only on matches before the test window.
It fits one model for home scoring and one model for away scoring. Each model
uses:

- an intercept
- attacking-team effects
- defending-team effects

The default target is `goals`, because Poisson regression naturally models count
outcomes. Experimental `xg` and `np_xg` targets are supported when those columns
are present, but the default comparison config uses goals.

Expected goals are capped to a sensible range, defaulting to `0.05` through
`5.0`. The model then builds an independent Poisson score grid through
`max_goals`, defaulting to 8, and sums scoreline probabilities into:

- `p_home_win`
- `p_draw`
- `p_away_win`

The probabilities are normalised so each prediction row sums to 1.

## Regularisation and shrinkage

Team attack and defence effects are ridge-regularised. The intercept is not
penalised, so the league scoring baseline remains free to fit the fold. Ridge
regularisation keeps tiny-sample team effects closer to neutral and helps avoid
unstable extreme rates from one or two unusual matches.

Unseen teams at prediction time receive zero attack and defence effects, which
means they use the learned league-average baseline for the relevant home or away
scoring model.

## Running a comparison

Use the local comparison runner. It reads a CSV and does not require live
Supabase access.

```powershell
.\.venv\Scripts\python.exe scripts\run_model_comparison.py --csv data\exports\wsl_match_data.csv --model champion_dc_xg --model regularised_team_strength --model poisson_regression --model logistic_regression --model elo_baseline --model naive_outcome_rate --test-start 2025-10-01 --test-end 2026-05-16 --min-train-matches 12 --output-md reports/poisson_regression_comparison_first_run.md --output-json reports/poisson_regression_comparison_first_run.json
```

Lower Brier score and log loss are better. Higher accuracy is better, but should
be interpreted alongside calibration-sensitive metrics.

## Output columns

The challenger emits the normal evaluation columns plus:

- `expected_home_goals`
- `expected_away_goals`
- `target_source`
- `training_matches`
- `home_fit_iterations`
- `away_fit_iterations`

## Current limitations

This is still a simple model. It does not include Dixon-Coles low-score
correlation, player availability, schedule congestion, market information, or
lineup context. With roughly one WSL season and about 109 evaluated matches in
the current comparison window, metric differences should be treated as
directional rather than definitive.
