# Model Evaluation Readiness Audit

Last audited: 2026-06-19

This is an audit-only report. It does not propose or apply code changes to the current model behavior.

## Executive Summary

The repository already has a serious foundation for industry-grade WSL model evaluation:

- A FastAPI prediction service backed by Supabase RPC data.
- A current champion model in `model/wsl_xg_model.py`.
- Date-based prediction splits and walk-forward backtesting.
- Logged prediction replay evaluation for Matchweeks 2-22.
- Reusable Brier score, log loss, accuracy and calibration metrics.
- Persistence paths for `prediction_runs` and `evaluation_runs`.
- A dashboard replay manifest aligned to Supabase-derived 2025-26 WSL fixture windows.
- Monte Carlo simulation and an emerging market odds benchmark path.
- CI, dependency constraints, Docker and Azure Container Apps infrastructure.

The main gap is not basic evaluation capability. The gap is a clean experiment framework for repeatable champion-vs-challenger comparison across multiple model families, datasets, feature sets and market configurations. The current codebase is strong operationally, but model research concepts are still spread across API, model, evaluation, scripts and reports rather than expressed as a unified experiment architecture.

## Current Working Tree Note

The current branch is `codex/market-odds-benchmark`. At audit time, the working tree already contained uncommitted market benchmark changes plus one untracked Monte Carlo report:

- Modified: `README.md`
- Modified: `docs/evaluation_strategy.md`
- Modified: `evaluation/evaluate_logged_predictions.py`
- Untracked: `docs/market_benchmark_strategy.md`
- Untracked: `evaluation/market_benchmark.py`
- Untracked: `scripts/run_market_benchmark.py`
- Untracked: `tests/test_market_benchmark.py`
- Untracked: `reports/monte_carlo_test.md`

This report treats those files as present in the local repository, but the market benchmark should be considered "branch-local" until merged.

## 1. Current Repository Structure

Current high-level structure:

```text
api/                    FastAPI app and endpoint contracts
dashboard/              Streamlit analyst dashboard and matchweek manifest
data/                   Supabase RPC data access
docs/                   Architecture, deployment, testing, dashboard, evaluation and simulation docs
evaluation/             Metrics, walk-forward evaluation, logged replay, persistence, market benchmark
infra/                  Azure Container Apps Bicep infrastructure
model/                  Current xG/Dixon-Coles prediction model
reports/                Generated replay, evaluation and simulation reports
scripts/                Operational scripts and SQL setup scripts
simulation/             Monte Carlo league-table simulation
tests/                  Unit and smoke tests
.github/workflows/      CI workflow
```

Where current responsibilities live:

| Concern | Current location |
| --- | --- |
| Data ingestion | `data/supabase_client.py` |
| Raw model data contract | `model/wsl_xg_model.py::REQUIRED_COLS`, `data/supabase_client.py::EXPECTED_COLS` |
| Feature engineering | Mostly embedded in `model/wsl_xg_model.py` |
| Model training | `estimate_team_strengths()`, `estimate_penalty_rates()`, `fit_rho()` in `model/wsl_xg_model.py` |
| Prediction generation | `predict_fixtures()` in `model/wsl_xg_model.py`, API `/predict` in `api/main.py` |
| Scoreline probabilities | `compute_scoreline_matrix()`, `wdl_from_matrix()`, `top_scorelines()` |
| Evaluation metrics | `evaluation/metrics.py` and `run_backtest()` in `model/wsl_xg_model.py` |
| Walk-forward evaluation | `model/wsl_xg_model.py::run_backtest()`, `evaluation/run_evaluation.py` |
| Logged replay evaluation | `evaluation/evaluate_logged_predictions.py` |
| Market benchmark | `evaluation/market_benchmark.py`, `scripts/run_market_benchmark.py` |
| Monte Carlo simulation | `simulation/monte_carlo.py`, `scripts/run_monte_carlo_simulation.py` |
| Prediction audit trail | `evaluation/eval_store.py`, `scripts/setup_prediction_runs_table.sql` |
| Evaluation audit trail | `evaluation/evaluation_store.py`, `scripts/setup_evaluation_runs_table.sql` |
| Dashboard replay | `dashboard/app.py`, `dashboard/matchweek_manifest.py`, dashboard components |
| Reporting outputs | `reports/*.md`, report builders in evaluation/simulation modules |
| Configuration | `ModelConfig`, request schemas, CLI flags, env vars, `requirements*.txt`, `constraints.txt` |

Central files for the current model:

- `model/wsl_xg_model.py`
- `data/supabase_client.py`
- `api/main.py`
- `evaluation/metrics.py`
- `evaluation/run_evaluation.py`
- `evaluation/evaluate_logged_predictions.py`
- `dashboard/matchweek_manifest.py`
- `reports/logged_replay_evaluation_2025_26.md`
- `reports/replay_manifest_check.md`

There are no notebooks in the repository. There are no committed raw CSV datasets beyond generated reports and SQL scripts.

## 2. Current Model Pipeline

### End-to-end workflow

1. Supabase exposes WSL match data through `rpc_wsl_weekly_stats()`.
2. `data/supabase_client.py::fetch_match_data()` validates and coerces the RPC result into a DataFrame.
3. API `/predict` receives `train_before`, `predict_from`, `predict_to` and model hyperparameters.
4. `split_played_future()` trains on matches before `train_before` and selects prediction fixtures in the requested date window.
5. `estimate_team_strengths()` fits team attack and defence parameters using weighted ridge regression on log non-penalty xG.
6. `estimate_penalty_rates()` estimates per-team penalty xG rates with shrinkage.
7. `fit_rho()` optionally fits Dixon-Coles low-score correlation from training results.
8. `predict_fixtures()` computes expected goals, scoreline matrices, win/draw/loss probabilities and top scorelines.
9. The API returns predictions and team strengths.
10. `evaluation/eval_store.py::log_prediction_run()` stores the run in Supabase `prediction_runs`.

### Current model type

The current champion model is an xG-driven modified Dixon-Coles / Poisson scoreline model:

- Team strengths are estimated from non-penalty xG, not from goals.
- Expected goals are produced from log-linear attack/defence parameters.
- Penalty xG is added as an independent Poisson component.
- Dixon-Coles adjustment is applied to low-score open-play outcomes.
- W/D/L probabilities are derived from the scoreline matrix.

### Inputs handled today

Required match-level fields:

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

Additional inputs supported by newer modules:

- Market odds CSV fields in `evaluation/market_benchmark.py`.
- Dashboard replay manifest fields in `dashboard/matchweek_manifest.py`.
- Simulation fixture lambdas in `simulation/monte_carlo.py`.

### Prediction outputs

`FixturePrediction.to_dict()` returns:

- teams
- match date
- round label as `round`
- total xG and non-penalty xG values
- home/draw/away probabilities
- top scoreline strings
- optional bootstrap confidence intervals

Important: the API response rounds probabilities to percentages. Internally, `FixturePrediction` retains raw probability values and lambdas.

### As-of support and leakage risk

The pipeline supports as-of-date prediction through:

- `train_before`
- `predict_from`
- `predict_to`

The dashboard gives as-of-round operational behavior by mapping matchweeks to date windows, but the core model remains date-based.

Leakage posture:

- Training is mostly leakage-safe because `split_played_future()` uses `match_date < train_before` and requires valid historical xG.
- `run_backtest()` trains strictly before each weekly batch.
- Prediction windows may include rows whose actual goals/xG are present in the full historical DataFrame, but `predict_fixtures()` does not use actual goals or future xG from those fixture rows.
- Rescheduled fixtures create round/date ambiguity. This has already been mitigated in logged replay evaluation by filtering Week N predictions to `round_label == R<N>`.
- Week 1 still requires historical priors or a previous-season baseline; current-season-only training cannot support pre-season Week 1 predictions.

Main leakage risk to watch: future challenger feature builders must not accidentally compute season-to-date or rolling features using matches on or after the prediction date. The current code does not yet have a central feature-building abstraction enforcing this.

## 3. Data Availability And Data Quality

### What data exists

The repository itself does not contain committed raw historical match CSVs. Local `.env` exists but was not read. Data appears to be served primarily by Supabase RPC `rpc_wsl_weekly_stats()`.

The committed report `reports/replay_manifest_check.md` confirms live Supabase data for WSL 2025-26:

- R1-R22 exist.
- Each round has 6 fixtures.
- Each round has 6 completed fixtures.
- Matchweeks 2-22 provide 126 replay fixtures.
- R3, R14, R16, R20 and R21 contain long/rescheduled date windows.

The logged replay report confirms:

- 126 evaluated fixtures for Weeks 2-22.
- Brier score: 0.519150.
- Log loss: 0.891373.
- Accuracy: 0.611111.
- High-confidence calibration appears strong.

### Seasons, leagues and competitions

Observed from repository/report artifacts:

- WSL 2025-26 is present via Supabase and dashboard manifest.
- No committed local evidence of additional WSL seasons.
- No committed Premier League dataset.
- No committed multi-league schema or competition identifier.

### Is there enough WSL data for rolling backtests?

There is enough data for a first rolling/backtest framework on one WSL season, especially Weeks 2-22. However, one season is not enough for stable model selection among logistic regression, gradient boosting, neural networks and transfer-learning variants.

For serious challenger model validation, the project should acquire or expose multiple seasons, ideally:

- WSL match results and xG for at least 3-5 seasons.
- Team identifiers stable across name changes.
- Competition/season fields.
- Betting odds snapshots where market comparison is desired.

### Data fields needed for better evaluation

Already available:

- match date
- round label
- home/away teams
- goals
- xG and non-penalty xG

Needed or not yet standardised:

- canonical `season`
- canonical `competition`
- stable `match_id`
- stable `team_id`
- venue/home-away flags if neutral venues ever occur
- kickoff timestamp rather than date only
- prediction timestamp
- odds snapshot timestamp
- odds source/bookmaker/aggregator
- closing/opening odds distinction
- current model version or code version
- feature snapshot metadata

### Schema inconsistencies and risks

- `round_label` is used for replay/evaluation but not core splitting. This is good for rescheduled fixtures but requires explicit matching logic.
- API predictions use `round`; some evaluation code accepts both `round` and `round_label`.
- Prediction outputs store probabilities as rounded percentages; evaluation normalizes them back to probabilities.
- There is no central canonical match key. Current matching uses date, round label, home team and away team.
- Team names are string-matched with casefolding in evaluation, but there is no team-ID mapping.
- `docs/current_architecture.md` is stale; it says no CI workflow existed during an older audit, but `.github/workflows/ci.yml` is now present.

## 4. Existing Evaluation Capability

Existing evaluation components:

- `model/wsl_xg_model.py::run_backtest()`
- `evaluation/metrics.py`
- `evaluation/run_evaluation.py`
- `evaluation/evaluate_logged_predictions.py`
- `evaluation/evaluation_store.py`
- `evaluation/market_benchmark.py`
- `reports/logged_replay_evaluation_2025_26.md`
- tests under `tests/`

Metrics already supported:

- Brier score
- multiclass log loss
- outcome accuracy
- confidence calibration bins
- low/medium/high confidence buckets
- per-match logged replay results
- duplicate run warnings
- unmatched prediction/actual reporting
- strict round-label filtering for rescheduled replay windows
- model-vs-market Brier/log-loss skill scores in the market benchmark branch

Evaluation quality assessment:

- Strong for current operational replay of one completed season.
- Good for unit-level metric correctness.
- Good for logged prediction replay evidence.
- Not yet a full model research framework.

Gaps:

- No central experiment registry/config format.
- No common model interface for champion and challengers.
- No standard feature store/build function with as-of guarantees.
- No multi-season dataset abstraction.
- No fold manifest for rolling-origin splits.
- No champion version freeze artifact.
- No model comparison table across multiple model families.
- No statistical uncertainty around model comparisons.
- No failure-analysis taxonomy beyond worst misses.
- No formal calibration plots/curves as artifacts.
- No market blending optimizer yet.

## 5. Backtesting Readiness

The current pipeline can run leakage-safe date-based backtests:

- `run_backtest()` groups played match dates into weekly batches.
- Each batch trains only on matches before the batch start date.
- It skips batches with insufficient training rows.
- `evaluation/run_evaluation.py` wraps this in a reusable runner.

What needs to change for industry-grade backtesting:

- Add a canonical `BacktestFold`/fold manifest object.
- Support both date-based and round-based fold definitions.
- Store exact train/test match IDs per fold.
- Persist fold metadata and data snapshot metadata.
- Support multiple model classes behind one interface.
- Ensure every feature builder accepts an as-of timestamp or cutoff.
- Produce comparison outputs across models for identical folds.

Recommended design:

```text
evaluation/backtesting.py
  BacktestConfig
  BacktestFold
  build_rolling_folds()
  run_backtest_for_model()
  compare_model_results()

features/
  match_features.py
  team_form_features.py
  market_features.py

models/
  base.py
  champion_dc_xg.py
  baselines.py
  sklearn_logistic.py
  lightgbm_model.py

experiments/configs/
  champion_2025_26.yaml
  challenger_logistic_baseline.yaml
```

## 6. Champion Model Definition

The current champion model should be:

> The xG-driven modified Dixon-Coles / Poisson scoreline model implemented in `model/wsl_xg_model.py` with the production default configuration used by `/predict`.

Champion-defining files/functions:

- `model/wsl_xg_model.py::ModelConfig`
- `model/wsl_xg_model.py::split_played_future`
- `model/wsl_xg_model.py::estimate_team_strengths`
- `model/wsl_xg_model.py::estimate_penalty_rates`
- `model/wsl_xg_model.py::fit_rho`
- `model/wsl_xg_model.py::compute_scoreline_matrix`
- `model/wsl_xg_model.py::predict_fixtures`
- `api/main.py::PredictRequest`
- `api/main.py::predict`

Recommended freeze/versioning approach:

- Create `models/champion_dc_xg.py` as an adapter around the existing functions, not a rewrite.
- Create a `ChampionModelSpec` or YAML config containing default alpha, decay, rho, max goals, bootstrap setting and data contract.
- Record git SHA/code version in evaluation outputs.
- Save champion predictions per fold before adding challengers.
- Treat the current logged replay report as the first champion evidence artifact, but not as the only model-selection evidence.

## 7. Challenger Model Readiness

| Challenger | Already supported? | Required new files/modules | Data needed | Difficulty | Expected value |
| --- | --- | --- | --- | --- | --- |
| Naive baseline | No explicit model, easy to add | `models/baselines.py`, tests | results only; optional league home/draw/away rates | Low | High as sanity baseline |
| Elo-only model | Not present | `models/elo.py`, `features/elo_features.py` | historical results by date, stable team IDs | Medium | High interpretability, useful baseline |
| Current Poisson/DC xG model | Yes as champion | adapter only: `models/champion_dc_xg.py` | current xG schema | Low | Essential reference |
| Logistic regression | Not present | `models/sklearn_logistic.py`, feature builders | rolling team features, outcome labels, enough seasons | Medium | Good first ML challenger |
| LightGBM/XGBoost | Not present | `models/gbm.py`, dependency decision, configs | richer features, more historical data | Medium-high | Potentially high with more seasons |
| Simple neural network | Not present | `models/nn.py`, likely PyTorch dependency | much more data or strong regularisation | High | Low-medium initially; useful experiment later |
| Market-only baseline | Branch-local market benchmark exists, but no model adapter | `models/market.py`, odds data loader | odds snapshots for each fixture | Medium | Very high benchmark value |
| Model-plus-market blend | Not present | `models/blend.py`, calibration/blending optimizer | aligned model predictions, market probabilities, actuals | Medium | High if odds data exists |
| PL-trained / WSL-calibrated model | Not present | `data/multi_league_loader.py`, `features/league_features.py`, transfer configs | PL and WSL match/xG data, league labels, team mappings | High | Experimental; useful interview/research angle |

## 8. Recommended Target Architecture

Avoid a large packaging migration, but separate research concerns cleanly:

```text
data/
  supabase_client.py
  schemas.py
  loaders.py
  market_odds.py

features/
  base.py
  match_features.py
  team_form.py
  market_features.py

models/
  base.py
  champion_dc_xg.py
  baselines.py
  elo.py
  logistic.py
  gbm.py
  blend.py

evaluation/
  metrics.py
  backtesting.py
  calibration.py
  compare.py
  failure_analysis.py
  evaluate_logged_predictions.py
  market_benchmark.py
  evaluation_store.py

experiments/
  configs/
    champion_dc_xg.yaml
    naive_baseline.yaml
    logistic_v1.yaml
  registry.py

reports/
  model_comparison_*.md
  calibration_*.md
  failure_analysis_*.md

scripts/
  run_backtest_experiment.py
  run_model_comparison.py
  run_market_benchmark.py
```

Keep `model/wsl_xg_model.py` intact initially. Use an adapter so challenger work does not destabilize production.

## 9. Phased Implementation Roadmap

### Phase 1 - Audit fixes and data schema standardisation

Files to create/modify:

- `docs/model_evaluation_readiness_audit.md`
- `data/schemas.py`
- `docs/data_contract.md`
- possibly update stale `docs/current_architecture.md`

Acceptance criteria:

- Canonical match schema documented.
- Required/optional columns defined.
- Match key and season/competition conventions defined.
- No model behavior changes.

Expected outputs:

- Data contract doc.
- Schema validation helpers and tests.

Risks:

- Supabase RPC may not expose stable `season`, `competition`, `match_id` or team IDs yet.

Branch/PR:

- Yes. Separate first PR.

### Phase 2 - Current champion model evaluation

Files to create/modify:

- `models/base.py`
- `models/champion_dc_xg.py`
- `experiments/configs/champion_dc_xg.yaml`
- `evaluation/champion_runner.py`

Acceptance criteria:

- Champion adapter produces the same probabilities as the current model functions.
- Evaluation records include champion model name/config/version.

Expected outputs:

- Champion frozen config.
- Champion evaluation report.

Risks:

- Adapter drift if it copies logic instead of calling existing functions.

Branch/PR:

- Yes.

### Phase 3 - Rolling backtest framework

Files to create/modify:

- `evaluation/backtesting.py`
- `tests/test_backtesting.py`
- `scripts/run_backtest_experiment.py`

Acceptance criteria:

- Fold builder creates deterministic train/test splits.
- Each fold has train/test match identifiers or keys.
- Every model runs on identical folds.

Expected outputs:

- Fold manifest.
- Backtest result JSON/Markdown.

Risks:

- Without stable match IDs, keys remain string/date based.

Branch/PR:

- Yes.

### Phase 4 - Metric and calibration report

Files to create/modify:

- `evaluation/calibration.py`
- `evaluation/compare.py`
- `evaluation/failure_analysis.py`
- report templates

Acceptance criteria:

- Metrics include Brier, log loss, accuracy, calibration bins and confidence buckets.
- Report includes best/worst fixtures and calibration diagnostics.

Expected outputs:

- `reports/champion_evaluation_*.md`
- optional JSON artifact.

Risks:

- Small sample sizes can make calibration noisy.

Branch/PR:

- Yes.

### Phase 5 - Challenger baseline models

Files to create/modify:

- `models/baselines.py`
- `models/elo.py`
- `models/logistic.py`
- `features/team_form.py`
- configs for each challenger

Acceptance criteria:

- Naive, Elo and logistic regression challengers run on the same folds as champion.
- Model comparison table ranks by log loss and Brier score.

Expected outputs:

- Model comparison report.
- Challenger prediction artifacts.

Risks:

- One WSL season is too small for robust ML conclusions.

Branch/PR:

- Yes, likely split naive/Elo and logistic into separate PRs.

### Phase 6 - Market blending

Files to create/modify:

- `data/market_odds.py`
- `models/market.py`
- `models/blend.py`
- `evaluation/market_benchmark.py`

Acceptance criteria:

- Market-only baseline runs as a model.
- Blend weights are learned only on training folds.
- Market/model skill is reported.

Expected outputs:

- Market benchmark report.
- Blend comparison report.

Risks:

- Odds snapshot timing and market liquidity can dominate conclusions.

Branch/PR:

- Yes.

### Phase 7 - PL-to-WSL transfer/adaptation experiments

Files to create/modify:

- `data/multi_league_loader.py`
- `features/league_features.py`
- transfer/adaptation model configs

Acceptance criteria:

- PL-trained features/models can be evaluated on WSL folds without leakage.
- League indicator/calibration layer tested.

Expected outputs:

- Transfer experiment report.

Risks:

- Domain shift between men's PL and WSL may make transfer weak or misleading.
- Requires external data acquisition and licensing clarity.

Branch/PR:

- Yes, separate research branch.

### Phase 8 - Final pre-season model selection report

Files to create/modify:

- `reports/preseason_model_selection_YYYY.md`
- `docs/model_selection_decision.md`

Acceptance criteria:

- Champion and challengers compared on identical leakage-safe folds.
- Recommendation documented with metrics, calibration, failure analysis and operational risk.

Expected outputs:

- Pre-season model selection report.
- Recommended production champion config.

Risks:

- Small WSL sample may favor conservative champion retention.

Branch/PR:

- Yes.

## 10. Immediate Next PR Recommendation

Smallest safe first PR:

> Add a data contract and experiment-readiness scaffold without changing model behavior.

Recommended branch:

`codex/evaluation-data-contract-baseline`

Recommended first PR scope:

- Add `docs/data_contract.md`.
- Add `data/schemas.py` with constants for canonical column names and lightweight validation helpers.
- Add tests for schema validation using mocked DataFrames.
- Add `docs/model_evaluation_readiness_audit.md` if not already merged.
- Do not alter `model/wsl_xg_model.py`, API responses, dashboard behavior or prediction outputs.

Acceptance criteria:

- Existing tests still pass.
- New schema tests pass.
- The champion model remains behaviorally identical.
- The data contract clearly distinguishes required model columns, evaluation columns, market columns and future multi-league columns.

## What Is Already Done

- Production-style FastAPI prediction API.
- Supabase RPC ingestion.
- Current xG/Dixon-Coles champion model.
- Date-based splitting and walk-forward backtesting.
- Reusable Brier, log loss, accuracy and calibration metrics.
- Prediction run logging to `prediction_runs`.
- Offline evaluation persistence to `evaluation_runs`.
- Logged replay evaluation for dashboard-generated predictions.
- Verified 2025-26 WSL replay manifest.
- Strict round-label evaluation for rescheduled rounds.
- Monte Carlo simulation baseline.
- Market benchmark module in the local branch.
- CI, Docker, dependency constraints and Azure infrastructure.

## What Is Missing

- Multi-season WSL dataset in the repository or exposed schema.
- Stable match/team IDs and canonical season/competition fields.
- Central feature-building layer with as-of guarantees.
- Unified model interface.
- Champion model adapter/frozen config.
- Challenger model registry.
- Rolling fold manifest persisted as an artifact.
- Model comparison report across champion/challengers.
- Calibration/failure-analysis report as first-class artifact.
- Market blending model and fold-safe blend optimization.
- PL-to-WSL data integration.

## Main Risks

- Single-season WSL sample size may be too small for high-capacity challengers.
- String-based matching can fail under team name changes.
- Rescheduled fixtures require careful date/round handling.
- Full-season Supabase data can tempt feature leakage if future feature builders are not as-of constrained.
- Market odds conclusions depend heavily on odds source and snapshot timing.
- Documentation drift already exists in older architecture docs.

## Recommended First Implementation Branch

`codex/evaluation-data-contract-baseline`

## Recommended First PR Scope

Add the canonical data contract, schema validators and tests. This is the safest move toward champion-vs-challenger evaluation because it reduces leakage and matching risk before adding any new model family.
