# First Model Comparison Runbook

## Status

The first real champion-vs-challenger model comparison was not run in this
branch because no suitable local WSL match-data export is currently committed
to the repository.

This is intentional: the report must be based on real historical match rows,
not reconstructed or fabricated data.

## Local Data Inspection

Checked local repository data sources:

- `data/` contains schema/client code only, not a match CSV export.
- `reports/logged_replay_evaluation_2025_26.md` contains aggregated logged
  champion replay metrics plus diagnostic JSON blocks, but not the full
  training/evaluation dataset required to rerun all models.
- `reports/replay_manifest_check.md` contains fixture-window metadata only.
- No committed `.csv`, `.json`, `.parquet`, or `.xlsx` WSL match export exists
  outside the virtual environment.

Because the champion adapter requires xG columns and the challengers require
completed match result rows, the existing Markdown reports are insufficient for
a real four-model rolling comparison.

## Required Export

Create a local CSV export such as:

`data/exports/wsl_2025_26_match_data.csv`

Required columns for all comparison models:

- `match_date`
- `round_label`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`

Additional columns required for `champion_dc_xg`:

- `home_xg`
- `away_xg`
- `home_np_xg`
- `away_np_xg`

Recommended optional columns for auditability:

- `season`
- `competition`
- `match_id`

The export should include completed historical fixtures with final scores and
available xG values. Dates must be the actual played dates, including postponed
or rearranged fixtures.

## Command To Run

Once the CSV exists, run:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_comparison.py `
  --csv data\exports\wsl_2025_26_match_data.csv `
  --model champion_dc_xg `
  --model naive_outcome_rate `
  --model elo_baseline `
  --model logistic_regression `
  --test-start 2025-09-12 `
  --test-end 2026-05-16 `
  --test-window-days 7 `
  --step-days 7 `
  --min-train-matches 6 `
  --output-md reports\model_comparison_first_run.md `
  --output-json reports\model_comparison_first_run.json
```

If the CSV lacks xG columns, run the challenger-only comparison first:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_comparison.py `
  --csv data\exports\wsl_2025_26_match_data.csv `
  --model naive_outcome_rate `
  --model elo_baseline `
  --model logistic_regression `
  --test-start 2025-09-12 `
  --test-end 2026-05-16 `
  --test-window-days 7 `
  --step-days 7 `
  --min-train-matches 6 `
  --output-md reports\model_comparison_first_run.md `
  --output-json reports\model_comparison_first_run.json
```

## Expected Artefacts

The real run should produce:

- `reports/model_comparison_first_run.md`
- `reports/model_comparison_first_run.json`

The Markdown report should include:

- dataset used
- date range and fold settings
- models compared
- Brier score, log loss, and accuracy
- calibration and confidence summaries
- worst misses and high-confidence correct calls
- limitations and next recommendations

## Leakage Safety

The runner builds rolling folds once and reuses the same folds for every model.
For each fold, training rows must have `match_date < test_start`; test-window
matches are never included in feature construction or model fitting.

## Current Limitations

- The repository currently stores evaluation reports, not the source match-data
  export needed to reproduce the first real comparison.
- Champion comparison requires xG fields; result-only fixture exports can only
  evaluate the naive, Elo, and logistic challengers.
- WSL sample sizes are small, so early-season folds may need
  `--min-train-matches 6` or a similar threshold.

