# Streamlit Operations Dashboard Strategy

## Purpose

The Streamlit dashboard is the internal analyst control centre for weekly WSL prediction operations. It removes the need to manually assemble date windows, `curl` commands, and evaluation commands during normal matchweek workflows.

This dashboard is not the prediction engine. FastAPI remains the authoritative service for model execution, prediction logging, authentication, and production API behaviour.

## Architecture

```text
Analyst browser
      |
      v
Streamlit dashboard
      |
      | POST /predict, GET /history
      v
FastAPI prediction engine
      |
      v
Supabase data and audit tables
```

The baseline dashboard runs locally and points to the backend through `PREDICTION_API_BASE_URL`, defaulting to `http://localhost:8000`.

## Local Run Steps

Install dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt
```

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

The dashboard sidebar lets an analyst set:

- API base URL
- API key
- Season
- Matchweek

## Matchweek Manifest

The file `dashboard/matchweek_manifest.py` contains the local season manifest. Each matchweek maps to:

- `train_before`
- `predict_from`
- `predict_to`
- `status`
- `note`
- `verified`

The replay-readiness baseline includes all 22 WSL 2025-26 matchweek entries. The dates are still placeholders unless `verified=True`, so every window must be checked against official WSL fixtures before final replay.

## Full-Season Replay Readiness Workflow

The operational replay target is Matchweeks 2-22 of the completed WSL 2025-26 season. Matchweek 1 is excluded from this baseline because it needs historical priors or a previous-season baseline before any current-season matches exist.

Before replaying, inspect the fixture windows exposed by the same Supabase RPC used by the prediction API:

```bash
python scripts/inspect_matchweek_windows.py --output reports/replay_manifest_check.md
```

To replay week by week:

1. Verify each Matchweek 2-22 manifest window against the data-derived fixture windows and official fixture list.
2. Set `verified=True` and update `status` or `note` once a window has been checked.
3. Start FastAPI locally with `.env` loaded.
4. Start Streamlit and select the season and matchweek.
5. Confirm the dashboard status labels before generating predictions.
6. Click **Generate Predictions** for the selected week.
7. Confirm each generated prediction creates a `prediction_runs` record by refreshing prediction history.
8. Run evaluation after predictions are logged.

The current status inference is intentionally simple. A matchweek is shown as `Predicted` when recent prediction history contains a matching `predict_from` and `predict_to` window. Evaluation status is not yet read reliably in the dashboard and is shown as unavailable/not evaluated.

## Week 1 Limitation

Matchweek 1 requires historical priors or a previous-season baseline because no current-season matches exist before the first fixture. This branch only displays that operational warning. It does not implement historical-prior modelling.

Monte Carlo simulation is also deferred until baseline replay metrics are captured.

## Prediction Workflow

When an analyst clicks **Generate Predictions**, Streamlit calls `POST /predict` on FastAPI with:

- `train_before`
- `predict_from`
- `predict_to`
- `run_trigger`

The generated `run_trigger` follows:

```text
dashboard-season-2025-26-week-07
```

The dashboard displays:

- `run_id`
- training match count
- prediction fixture count
- rho used
- prediction table
- team strength table

## Prediction History

The baseline history panel calls `GET /history?n=10` through FastAPI. This keeps prediction history access behind the same API key flow as prediction generation.

The main history table is flattened so nested prediction payloads do not render as raw objects. Detailed run JSON remains available in an expandable preview.

`dashboard/supabase_reader.py` exists for future direct-read panels such as persisted evaluation history. It uses environment variables and does not print secrets.

`prediction_runs` and `evaluation_runs` are separate audit paths:

- `prediction_runs` records generated prediction windows and prediction payloads from FastAPI.
- `evaluation_runs` records offline evaluation/backtest results when the evaluation runner is executed with `--persist`.

## Evaluation Workflow

The baseline evaluation panel shows local commands for the selected matchweek:

```bash
python -m evaluation.run_evaluation --start-date 2025-10-03 --run-trigger dashboard-season-2025-26-week-05-evaluation
```

If the `evaluation_runs` table is configured, the analyst can persist the result:

```bash
python -m evaluation.run_evaluation --start-date 2025-10-03 --run-trigger dashboard-season-2025-26-week-05-evaluation --persist
```

The current runner may evaluate from the selected start date onward unless a tighter matchweek-only window is implemented later. Automated evaluation execution from Streamlit is deferred so this checkpoint does not add new local process execution behaviour to the dashboard.

## Current Limitations

- Manifest dates are placeholders until individually verified.
- Matchweek 1 historical-prior support is only documented and warned about.
- Monte Carlo simulation is deferred until after baseline replay metrics are captured.
- Evaluation status is not inferred from `evaluation_runs` in the dashboard yet.
- Dashboard-triggered evaluation execution is deferred to a future branch.
- The dashboard is still an internal local analyst tool, not a public frontend.

## Azure Path

When FastAPI is hosted in Azure Container Apps, set:

```bash
PREDICTION_API_BASE_URL=https://<prediction-api-host>
```

The Streamlit dashboard can then target the hosted FastAPI endpoint without changing dashboard code. A later checkpoint should decide whether Streamlit is hosted separately, kept local-only, or placed behind an internal access gateway.

## Security Notes

- The dashboard is an internal analyst tool, not a public frontend.
- API keys are entered through a masked sidebar input.
- Secrets remain in `.env` locally and Azure Key Vault for production services.
- Unit tests use mocked HTTP clients and do not require live Supabase or live FastAPI.
