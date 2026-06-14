# Enterprise Roadmap

Last audited: 2026-06-13

This roadmap captures enterprise hardening gaps found in the repository audit. It is intentionally split into small checkpoints so changes stay reviewable and do not alter the core model logic until there is a clear reason.

## Current Baseline

Already present:

- FastAPI service with prediction, strengths, backtest, history, health, and readiness endpoints.
- Supabase RPC data ingestion through `rpc_wsl_weekly_stats()`.
- Existing Dixon-Coles/xG model layer.
- Supabase audit table script for prediction runs.
- Dockerfile targeting Python 3.11.
- Azure Container Apps Bicep infrastructure with Key Vault and Application Insights resources.
- Unit tests for schema validation, model invariants, API auth failures, and mocked API smoke paths.
- Python 3.11 development dependency baseline in `requirements-dev.txt`.
- Pytest and Ruff configuration in `pyproject.toml`.
- GitHub Actions CI baseline for linting, tests, security checks, and Docker build validation.
- `.gitignore` coverage for `.env`, raw data folders, generated predictions, and CSV files.

## Priority Gaps

### P0: Baseline Safety And Reproducibility

- Add secret scanning to CI.
- Add an explicit Python version file so local shells, CI, and Docker all agree on Python 3.11.
- Pin development/test dependency versions once the baseline package set stabilizes.
- Add a pre-commit or documented local quality gate for formatting, linting, and secret detection.
- Verify `.env` and private data files are untracked before every release.

### P1: API Hardening

- Replace broad default CORS behavior with an explicit production allow-list.
- Add structured request logging with correlation IDs.
- Add consistent error response schemas.
- Add request validation for date windows, maximum bootstrap size, and history result limits.
- Add rate limiting or upstream gateway policy for public ingress.
- Decide whether `/ready` should remain unauthenticated because it currently exercises Supabase data access.

### P1: Deployment And Secret Management

- Confirm Azure Container App managed identity has permission to read Key Vault secrets. Add the role assignment to Bicep if missing.
- Add deployment parameters examples that do not include secret values.
- Add post-deployment smoke test steps for `/health`, `/ready`, and an authenticated dry-run prediction window.
- Add rollback steps using Container Apps revisions.
- Configure Application Insights ingestion in the app runtime. The dependency exists, and the connection string is injected, but no instrumentation setup was found in application code.

### P1: Data Contract And Supabase Reliability

- Add a contract test or validation script for `rpc_wsl_weekly_stats()`.
- Document ownership, expected columns, data types, nullability, and freshness expectations for the RPC.
- Add defensive handling for empty future windows and insufficient training data at the API boundary.
- Add monitoring for RPC failure rate and response size anomalies.
- Decide whether the production API should use service role credentials directly or a narrower Supabase credential/path.

### P2: Evaluation And Audit Trail

- Persist `/backtest` runs or create a separate `backtest_runs` audit table.
- Add model version metadata to prediction records.
- Add code revision, image tag, and deployment environment to audit records.
- Add scheduled prediction run metadata once scheduling exists.
- Add retention and export policy for prediction audit data.

### P2: Test Coverage

- Add API tests for `/strengths`, `/history`, readiness behavior, and validation edge cases.
- Add regression fixtures for representative WSL data without private exports.
- Add Docker runtime smoke tests against `/health`.
- Add Bicep validation in CI.

### P3: Operational Maturity

- Define SLOs for availability, prediction latency, and data freshness.
- Add dashboards for request volume, failures, latency, Supabase RPC status, and prediction run counts.
- Add alerts for failed predictions, missing data, deployment failure, and high 5xx rate.
- Add incident response steps and owner contacts.
- Add cost guardrails for Azure resources.

## Recommended Checkpoints

1. `codex/enterprise-docs-baseline`: documentation-only baseline from this audit.
2. `codex/ci-test-baseline`: add GitHub Actions, test dependencies, linting, and secret scanning.
3. `codex/api-hardening-baseline`: request validation, CORS tightening, error schemas, and correlation IDs.
4. `codex/azure-deployment-hardening`: Key Vault access verification, Bicep validation, smoke tests, rollback docs.
5. `codex/evaluation-audit-v2`: model versioning, backtest persistence, and richer audit metadata.

## Non-Goals For This First Phase

- Do not change `model/wsl_xg_model.py` core prediction logic unless a clear bug is found.
- Do not remove existing endpoints.
- Do not commit local secrets, raw exports, private CSVs, or generated prediction files.
- Do not change the target stack: Python 3.11, FastAPI, Supabase, Docker, Azure Container Apps.
