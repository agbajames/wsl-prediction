# Current Architecture

Last audited: 2026-06-13

This document records what is currently present in the repository, based on a code audit only. It does not describe planned behavior unless explicitly marked as a gap.

## Repository Structure

```text
.
├── api/
│   └── main.py
├── data/
│   └── supabase_client.py
├── evaluation/
│   └── eval_store.py
├── infra/
│   └── container_app.bicep
├── model/
│   └── wsl_xg_model.py
├── scripts/
│   └── setup_prediction_runs_table.sql
├── tests/
│   └── test_prediction_pipeline.py
├── Dockerfile
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

The local `.env` file is present for development and is ignored by git. Raw CSV-style data is also ignored through `raw/`, `predictions/`, and `*.csv`.

## Runtime Flow

```text
Supabase rpc_wsl_weekly_stats()
        |
        v
data/supabase_client.py
        |
        v
model/wsl_xg_model.py
        |
        v
api/main.py
        |
        v
evaluation/eval_store.py -> Supabase prediction_runs
```

## API Layer

Implemented in `api/main.py` with FastAPI.

Endpoints:

| Method | Path | Purpose | Auth |
| --- | --- | --- | --- |
| GET | `/health` | Liveness check | None |
| GET | `/ready` | Supabase readiness check through `fetch_match_data()` | None |
| POST | `/predict` | Generate predictions for a date window and log successful runs | `X-API-Key` |
| GET | `/strengths` | Return team attack/defence strength ratings as of a training cutoff | `X-API-Key` |
| POST | `/backtest` | Run walk-forward backtest on demand | `X-API-Key` |
| GET | `/history` | Return recent prediction run audit records | `X-API-Key` |

Authentication is a single API key read from the `API_KEY` environment variable. CORS is configured from `ALLOWED_ORIGINS`, defaulting to `*` if the variable is absent.

## Data Ingestion

Implemented in `data/supabase_client.py`.

The data layer calls Supabase RPC `rpc_wsl_weekly_stats()` and converts the returned rows into a Pandas DataFrame. It validates the expected columns against the model contract:

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

Secrets are read from:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

## Model Layer

Implemented in `model/wsl_xg_model.py`.

The model module contains the existing xG-driven Dixon-Coles prediction logic, including:

- CSV validation and loading for the standalone CLI path.
- Train/future split helpers.
- Team attack/defence estimation with ridge regularisation and time decay.
- Penalty xG rate estimation.
- Dixon-Coles scoreline matrix construction.
- Win/draw/loss probability generation.
- Optional bootstrap confidence intervals.
- Walk-forward backtesting with Brier score, log loss, and calibration bins.

This first hardening phase intentionally leaves the core model logic unchanged.

## Evaluation And Audit Trail

Implemented in `evaluation/eval_store.py` and `scripts/setup_prediction_runs_table.sql`.

Successful `/predict` calls write a record to the Supabase `prediction_runs` table with:

- Prediction window.
- Model configuration.
- Predictions.
- Team strengths.
- Rho value used.
- Optional backtest metrics.
- Run trigger.

The `/history` endpoint reads recent records from the same table.

The standalone `/backtest` endpoint returns metrics but currently does not persist a backtest-only audit record.

## Docker

Implemented in `Dockerfile`.

Current characteristics:

- Base image: `python:3.11-slim`.
- Installs dependencies from `requirements.txt`.
- Copies `data/`, `model/`, `api/`, and `evaluation/`.
- Runs as non-root user `appuser`.
- Starts Uvicorn on `${PORT:-8000}` with two workers.

## Azure Infrastructure

Implemented in `infra/container_app.bicep`.

Current resources:

- Log Analytics workspace.
- Application Insights component.
- Azure Key Vault.
- Key Vault secrets for Supabase URL, Supabase key, API key, and Application Insights connection string.
- Azure Container Apps environment.
- Azure Container App with external HTTP ingress, system-assigned identity, secret references, and scale-to-zero.

The Bicep file defines Key Vault references from the Container App, but the audit did not find an explicit role assignment granting the app identity access to Key Vault secrets.

## Tests

Implemented in `tests/test_prediction_pipeline.py`.

Current coverage includes:

- Expected Supabase schema matching model required columns.
- Train/future split behavior.
- Core probability invariants.
- Team strength constraints.
- Prediction generation over fixtures.
- Health endpoint behavior with mocked Supabase startup.

The repository does not currently include integration tests against a real Supabase instance, contract tests for `rpc_wsl_weekly_stats()`, Docker image tests, or deployment smoke tests.

## CI/CD

No `.github/workflows/` directory was found in the repository during this audit.

The README says pushes to `main` are handled by GitHub Actions, but the workflow files are not currently present in the tracked project.

## Documentation

Existing documentation before this baseline:

- `README.md`
- Inline module docstrings.
- SQL comments in `scripts/setup_prediction_runs_table.sql`.
- Bicep comments in `infra/container_app.bicep`.

Added by this baseline:

- `docs/current_architecture.md`
- `docs/enterprise_roadmap.md`
- `docs/deployment_runbook.md`
