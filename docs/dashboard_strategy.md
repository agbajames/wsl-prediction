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
uvicorn api.main:app --reload
```

Start the dashboard in a second terminal:

```bash
PREDICTION_API_BASE_URL=http://localhost:8000 \
API_KEY=local-dev-api-key \
streamlit run dashboard/app.py
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
- `notes`

The first baseline includes placeholder early 2025-26 windows so the structure is ready for the full 22-week fixture list. Dates should be verified against the official WSL schedule before operational use.

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

`dashboard/supabase_reader.py` exists for future direct-read panels such as persisted evaluation history. It uses environment variables and does not print secrets.

## Evaluation Workflow

The baseline evaluation panel shows local commands for the selected matchweek:

```bash
python -m evaluation.run_evaluation --start-date 2025-10-03 --run-trigger dashboard-season-2025-26-week-05
```

If the `evaluation_runs` table is configured, the analyst can persist the result:

```bash
python -m evaluation.run_evaluation --start-date 2025-10-03 --run-trigger dashboard-season-2025-26-week-05 --persist
```

Automated evaluation execution from Streamlit is deferred so this checkpoint does not add new local process execution behaviour to the dashboard.

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
