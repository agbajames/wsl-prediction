# WSL Match Data Export

This export tool writes the real WSL match-level dataset from Supabase to a
local CSV so the champion and challenger models can be evaluated on identical
rolling folds.

## Data Source

The script uses the existing data layer in `data/supabase_client.py`:

- `get_supabase_client()` creates a Supabase client from environment variables.
- `fetch_match_data()` calls `rpc_wsl_weekly_stats()`.
- The RPC result is validated and coerced to the schema expected by the current
  champion model.

## Required Environment Variables

Set these locally before running the export:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

Do not commit `.env` or print these values. The export script prints only a
safe summary: row count, date range, round labels, and output path.

## Export Command

```powershell
.\.venv\Scripts\python.exe scripts\export_wsl_match_data.py `
  --output data\exports\wsl_match_data.csv
```

The output directory is created if it does not already exist.

## Required Columns

The exported CSV must include:

- `match_date`
- `round_label`
- `home_team`
- `away_team`
- `home_xg`
- `away_xg`
- `home_np_xg`
- `away_np_xg`
- `home_goals`
- `away_goals`

The script fails clearly if any required column is missing.

## Use With Model Comparison

Once exported, run:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_comparison.py `
  --csv data\exports\wsl_match_data.csv `
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

## Commit Policy

Exported CSVs should stay local unless a future decision explicitly creates a
versioned public fixture. The repository ignores `*.csv` already, and
`data/exports/*.csv` is also ignored for clarity.

