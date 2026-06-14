# Dependency Management

Last updated: 2026-06-14

This project targets Python 3.11. The dependency strategy is intentionally lightweight:

- `requirements.txt` contains top-level production dependencies.
- `requirements-dev.txt` contains top-level development, test, lint, and audit dependencies.
- `constraints.txt` contains the resolved Python 3.11 dependency graph used to make local, CI, and Docker installs more stable.
- `.python-version` records the expected local Python family for tools such as pyenv.

## Install

Use the constraints file for local development:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt
```

Production installs should use:

```bash
python -m pip install -r requirements.txt -c constraints.txt
```

The Dockerfile follows the production install path.

## Updating Dependencies

For ordinary development, change only top-level dependencies in `requirements.txt` or `requirements-dev.txt`.

When dependency versions need to be refreshed:

```bash
python3.11 -m venv /tmp/wsl-prediction-constraints-venv
/tmp/wsl-prediction-constraints-venv/bin/python -m pip install --upgrade pip
/tmp/wsl-prediction-constraints-venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
/tmp/wsl-prediction-constraints-venv/bin/python -m pip freeze | sort -f > constraints.txt
```

Then run the validation gates:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt
python -m pytest tests/ -v
python -m ruff check .
python -m bandit -r api data evaluation model
python -m pip_audit
docker build -t wsl-prediction-engine:test .
```

## Security

`pip-audit` is blocking in CI. CI installs with `constraints.txt` first, then audits the installed environment so the result reflects the same resolved package graph used by tests.

If a security update requires changing a top-level dependency, update the top-level pin first, regenerate `constraints.txt`, and rerun the full validation gate.

## Non-Goals

This checkpoint does not introduce Poetry, Pipenv, uv, or a full packaging migration. A stronger lockfile tool can be considered later if release reproducibility needs hashes, multiple platform locks, or automated dependency update workflows.
