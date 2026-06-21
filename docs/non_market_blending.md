# Non-Market Blending Experiments

Phase 8D-1 adds offline fixed-weight ensemble experiments over existing model
probability outputs. The goal is to test whether simple averaging across
independent non-market model families improves probability scoring, log loss or
calibration enough to justify deeper investigation.

## Offline Only

This work does not change production prediction behaviour. The operational
reference remains `champion_dc_xg`, and `model/wsl_xg_model.py` is untouched.
The blend runner uses the existing rolling backtest framework, fits each
component model inside the same leakage-safe folds, normalises H/D/A
probabilities to unit scale, and then averages probabilities with fixed weights.

## Components

The default component set is:

- `champion_dc_xg`: unchanged operational/reference champion.
- `dc_fit_rho_each_fold`: best Phase 8A-3 champion-family probability-scoring variant.
- `txg_xg_pseudocount_010`: Phase 8A-4 conservative xG shrinkage reference.
- `regularised_team_strength`: strongest standalone non-champion statistical challenger.
- `improved_logistic_regression`: Phase 8B linear feature challenger.
- `random_forest`: Phase 8B tree-based challenger.
- `neural_network`: Phase 8C proof-of-concept research challenger.

The Phase 8A variants are included through their offline variant providers, not
through production routing.

## Predeclared Blend Grid

The grid is deliberately small:

- `blend_champion_regularised_50_50`: equal blend of champion and regularised team strength.
- `blend_champion_regularised_70_30`: champion-led blend with regularised team strength.
- `blend_champion_improved_logistic_70_30`: champion-led blend with improved logistic regression.
- `blend_champion_random_forest_70_30`: champion-led blend with random forest.
- `blend_champion_regularised_improved_logistic_60_20_20`: champion-led three-model blend.
- `blend_dc_fit_txg_50_50`: small champion-family reference blend from Phase 8A candidates.

These weights are fixed before evaluation. This avoids a broad search over one
WSL season and keeps the experiment useful as a robustness check rather than an
overfit leaderboard exercise.

## Interpretation

Any improvement from this phase should be treated as a research signal only. A
blend that improves Brier score or log loss in one run should be checked across
later seasons, calibration slices, and failure cases before any promotion is
considered. Accuracy alone is not enough to replace the champion because the
main modelling objective is probability quality.

## Running Locally

```powershell
.\.venv\Scripts\python.exe scripts\run_non_market_blending_experiment.py `
  --csv data\exports\wsl_match_data.csv `
  --test-start 2025-10-01 `
  --test-end 2026-05-16 `
  --min-train-matches 12 `
  --output-md reports\non_market_blending_first_run.md `
  --output-json reports\non_market_blending_first_run.json
```
