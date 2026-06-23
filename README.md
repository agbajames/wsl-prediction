# WSL Prediction Engine

xG-driven Dixon-Coles match prediction API for the Women's Super League — enterprise grade.

## Architecture

```
Supabase (rpc_wsl_weekly_stats)
        ↓
  data/supabase_client.py     # Direct RPC → DataFrame, no CSV
        ↓
  model/wsl_xg_model.py       # Modified Dixon-Coles (unchanged)
        ↓
  api/main.py                 # FastAPI: /predict /strengths /backtest /history
        ↓
  evaluation/eval_store.py    # Logs every run → prediction_runs table
        ↓
  Azure Container Apps        # Containerised, scale-to-zero, Key Vault secrets
```

## Local Setup

```bash
# 1. Install
python -m pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt

# 2. Configure environment
cp .env.example .env
# Fill in SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, API_KEY

# 3. Create Supabase audit table (run once)
# Paste scripts/setup_prediction_runs_table.sql into Supabase SQL editor

# 4. Run locally
python -m uvicorn api.main:app --env-file .env

# 5. Test
pytest tests/ -v
```

## Local Operations Dashboard

The internal Streamlit dashboard provides an analyst control centre for weekly prediction runs.

Start the FastAPI backend:

```bash
python -m uvicorn api.main:app --env-file .env
```

Start the dashboard in a second terminal:

```bash
PREDICTION_API_BASE_URL=http://localhost:8000 \
API_KEY=$API_KEY \
python -m streamlit run dashboard/app.py
```

The dashboard uses the selected season and matchweek manifest entry to call `POST /predict`, then displays the run metadata, prediction table, team strengths, and recent prediction history.

## API Usage

All prediction endpoints require `X-API-Key` header.

### Generate predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "train_before": "2026-05-13",
    "predict_from": "2026-05-16",
    "predict_to":   "2026-05-16"
  }'
```

### Team strengths
```bash
curl "http://localhost:8000/strengths?train_before=2026-05-13" \
  -H "X-API-Key: your-key"
```

### Run backtest
```bash
curl -X POST http://localhost:8000/backtest \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"backtest_start": "2025-10-01"}'
```

### Run offline evaluation
```bash
python -m evaluation.run_evaluation --start-date 2025-10-01
```

Evaluate logged dashboard replay predictions without re-running the model:
```bash
python -m evaluation.evaluate_logged_predictions \
  --season 2025-26 \
  --start-week 2 \
  --end-week 22
```

Persist an offline evaluation run to Supabase:
```bash
python -m evaluation.evaluate_logged_predictions \
  --season 2025-26 \
  --start-week 2 \
  --end-week 22 \
  --persist \
  --run-trigger logged-replay-2025-26-weeks-02-22
```

Run the evaluation-only market-implied benchmark against a local ignored odds export:
```bash
python scripts/run_market_benchmark.py \
  --csv data/exports/wsl_results_probabilities_2025_2026.csv \
  --output-md reports/market_benchmark_2025_26.md \
  --output-json reports/market_benchmark_2025_26.json \
  --output-rows reports/market_benchmark_2025_26_rows.csv
```

This derives proportional no-vig probabilities from raw odds and treats them as
an external market probability reference only. The supplied de-vigged
probability columns are retained for diagnostics. This is not model training
data, does not create model features and should not be used to claim that the
model beats bookmakers.

### Run mid-season Monte Carlo simulation
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

This builds the actual table through Matchweek 11, reuses the existing model pathway to produce expected-goals lambdas for Matchweeks 12-22, and samples Poisson scorelines to estimate title, top-3, and top-4 probabilities.

### Prediction history
```bash
curl "http://localhost:8000/history?n=5" \
  -H "X-API-Key: your-key"
```

## Deployment

Push to `main` — GitHub Actions handles the rest:
1. Runs tests
2. Builds Docker image → pushes to ACR
3. Deploys to Azure Container Apps

```bash
# One-time infra setup
az deployment group create \
  --resource-group wsl-analytics-rg \
  --template-file infra/container_app.bicep \
  --parameters containerImage=wslanalytics.azurecr.io/wsl-prediction-engine:latest \
               supabaseUrl=$SUPABASE_URL \
               supabaseKey=$SUPABASE_SERVICE_ROLE_KEY \
               apiKey=$API_KEY
```

## What changed from the original script

| Before | After |
|--------|-------|
| `--csv raw/wsl_round_22.csv` | Direct Supabase RPC call |
| Manual CSV export from Supabase | Automatic on every `/predict` call |
| `print()` to stdout | JSON API response |
| No audit trail | Every run logged to `prediction_runs` table |
| Local only | Azure Container Apps, scale-to-zero |
| No auth | API key (`X-API-Key` header) |
| No monitoring | Application Insights |

The model logic (`wsl_xg_model.py`) is **completely unchanged**.
