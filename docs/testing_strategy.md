# Testing Strategy

Last updated: 2026-06-13

This project targets Python 3.11 and keeps unit tests independent from live Supabase credentials.

## Local Setup

Install runtime and development dependencies:

```bash
python3.11 -m pip install -r requirements.txt -r requirements-dev.txt
```

Run the unit test suite:

```bash
python3.11 -m pytest tests/ -v
```

Run linting:

```bash
python3.11 -m ruff check .
```

Run a local Docker build validation:

```bash
docker build -t wsl-prediction-engine:test .
```

## Test Boundaries

Unit tests use local fixtures and mocks for Supabase-dependent paths. They should not require:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- A live Supabase network connection
- Raw private CSV exports

The API tests patch the data fetch and audit logging functions at the FastAPI module boundary. This keeps the tests focused on request handling, authentication, and response shape without relying on production credentials.

## Current Coverage

The baseline suite covers:

- API health endpoint.
- API authentication failures for missing and invalid `X-API-Key`.
- Supabase RPC response schema validation using mocked client data.
- Core model probability invariants.
- Prediction pipeline smoke test with a local DataFrame fixture.
- `/predict` endpoint smoke test with mocked Supabase data and audit logging.
- `/backtest` endpoint smoke test with mocked Supabase data and mocked backtest result.

## CI Checks

The GitHub Actions workflow runs on pull requests and pushes to `main`.

It performs:

- Python 3.11 dependency installation.
- Ruff linting.
- Pytest.
- Bandit security scanning for application modules.
- Advisory pip dependency audit.
- Docker build validation.

The CI workflow uses placeholder environment variables for tests and does not print or require real secrets.

The pip audit step is advisory in this baseline because current pinned dependencies may need a separate compatibility review before security upgrades are applied.

## Future Additions

- Add coverage thresholds once the baseline stabilizes.
- Add Supabase contract tests gated behind explicit integration-test configuration.
- Add Bicep validation.
- Add Docker runtime smoke tests against `/health`.
- Add API tests for `/strengths`, `/history`, and validation edge cases.
