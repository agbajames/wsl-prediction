# Deployment Runbook

Last audited: 2026-06-13

This runbook describes the current deployment path for the WSL Prediction Engine and highlights manual checks that should be performed until CI/CD is added.

## Required Inputs

Local development uses `.env`. Production should use Azure Key Vault.

Required variables:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `API_KEY`

Optional variable:

- `ALLOWED_ORIGINS`
- `APPINSIGHTS_CONNECTION_STRING`

Never commit `.env`, Supabase keys, API keys, raw exports, private CSVs, or generated prediction files.

## Local Setup

```bash
python --version
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with local values. Keep the file untracked.

Create the Supabase audit table once by running:

```bash
scripts/setup_prediction_runs_table.sql
```

The SQL should be run in the Supabase SQL editor for the target project.

## Local Run

```bash
uvicorn api.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Readiness check:

```bash
curl http://localhost:8000/ready
```

Authenticated prediction check:

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "train_before": "2026-05-13",
    "predict_from": "2026-05-16",
    "predict_to": "2026-05-16"
  }'
```

## Test Gate

Run before building or deploying:

```bash
pytest tests/ -v
```

If `pytest` is unavailable, install test dependencies in the active environment before release. The current `requirements.txt` does not list `pytest`.

## Container Build

```bash
docker build -t wsl-prediction-engine:local .
```

Optional local container run:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  wsl-prediction-engine:local
```

Then repeat the health, readiness, and authenticated prediction checks.

## Azure Infrastructure Deployment

The repository contains `infra/container_app.bicep`.

Current Bicep resources:

- Log Analytics workspace.
- Application Insights component.
- Key Vault and secrets.
- Azure Container Apps environment.
- Azure Container App with external ingress and scale-to-zero.

Example deployment command:

```bash
az deployment group create \
  --resource-group wsl-analytics-rg \
  --template-file infra/container_app.bicep \
  --parameters containerImage=<acr-name>.azurecr.io/wsl-prediction-engine:<tag> \
               supabaseUrl="$SUPABASE_URL" \
               supabaseKey="$SUPABASE_SERVICE_ROLE_KEY" \
               apiKey="$API_KEY"
```

Use secure parameter handling in production. Do not store real secret values in parameter files committed to git.

## Post-Deployment Checks

Capture the Container App URL from the Bicep output or Azure portal.

```bash
curl https://<container-app-url>/health
curl https://<container-app-url>/ready
```

Run one authenticated prediction request:

```bash
curl -X POST https://<container-app-url>/predict \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "train_before": "2026-05-13",
    "predict_from": "2026-05-16",
    "predict_to": "2026-05-16",
    "run_trigger": "manual"
  }'
```

Verify:

- The response contains predictions and team strengths.
- A row was inserted into Supabase `prediction_runs`.
- Application logs are visible in Azure log streaming or Log Analytics.
- No secret values appear in logs.

## Rollback

Until automated release management exists, use Azure Container Apps revisions:

```bash
az containerapp revision list \
  --resource-group wsl-analytics-rg \
  --name wsl-prediction-engine
```

Restore traffic to a previous known-good revision:

```bash
az containerapp ingress traffic set \
  --resource-group wsl-analytics-rg \
  --name wsl-prediction-engine \
  --revision-weight <revision-name>=100
```

After rollback, repeat health, readiness, and authenticated prediction checks.

## Current Release Risks

- No CI/CD workflow is present in the repository, despite README deployment claims.
- `pytest` is not listed in `requirements.txt`, so a fresh environment may not be able to run tests without an extra install.
- Application Insights connection string is provisioned and injected, but application instrumentation is not currently configured in code.
- Key Vault secret references are defined, but explicit Container App identity access to Key Vault was not found in the Bicep file.
- `/ready` is unauthenticated and performs a Supabase data fetch.
- Default CORS behavior is permissive if `ALLOWED_ORIGINS` is not set.
