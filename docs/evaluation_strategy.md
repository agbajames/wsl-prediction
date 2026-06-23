# Evaluation Strategy

Last updated: 2026-06-14

The evaluation baseline makes the WSL Prediction Engine measurable without changing the core model logic in `model/wsl_xg_model.py`.

## Components

- `evaluation/metrics.py`: reusable three-way football prediction metrics.
- `evaluation/run_evaluation.py`: repeatable walk-forward evaluation runner.
- `evaluation/evaluate_logged_predictions.py`: evaluates prediction runs already logged from the dashboard replay.
- `evaluation/market_benchmark.py`: evaluation-only market-implied benchmark validation and reporting.
- `evaluation/evaluation_store.py`: optional Supabase persistence for offline evaluation runs.
- `model/wsl_xg_model.py`: existing walk-forward backtest implementation reused as the prediction engine.
- `evaluation/eval_store.py`: existing prediction-run audit logging, unchanged in this checkpoint.

## Metrics

The standard metric bundle includes:

- Brier score for home/draw/away probabilities.
- Multiclass log loss.
- Outcome accuracy from the highest-probability class.
- Confidence calibration bins.
- Low/medium/high confidence bucket summary.

Probability rows are validated before scoring. Invalid inputs such as negative probabilities, non-finite values, wrong column counts, or zero-mass rows raise `ValueError` with clear messages. Valid positive rows are normalized before metric calculation so minor probability-sum drift does not break evaluation.

## Runner

Manual walk-forward evaluation uses the existing Supabase data layer and re-runs the model:

```bash
python -m evaluation.run_evaluation --start-date 2025-10-01
```

Persist an offline evaluation run:

```bash
python -m evaluation.run_evaluation \
  --start-date 2025-10-01 \
  --persist \
  --run-trigger manual \
  --code-version "$(git rev-parse --short HEAD)"
```

The runner:

- Fetches match data through `data.supabase_client.fetch_match_data()` when no DataFrame is provided.
- Supports local or mocked DataFrames in tests.
- Calls the existing model `run_backtest()` function.
- Returns a structured JSON-style result with parameters, aggregate metrics, model backtest metrics, and per-match results.
- Writes to Supabase only when `--persist` or `persist=True` is explicitly used.

## Logged Prediction Evaluation

For the dashboard replay, logged prediction evaluation is the correct evidence path because the predictions were already generated and stored in `prediction_runs`. It evaluates exactly what the dashboard produced instead of re-running the model with potentially different data or parameters.

Evaluate the full Week 2-22 replay:

```bash
python -m evaluation.evaluate_logged_predictions \
  --season 2025-26 \
  --start-week 2 \
  --end-week 22
```

Persist the result to `evaluation_runs`:

```bash
python -m evaluation.evaluate_logged_predictions \
  --season 2025-26 \
  --start-week 2 \
  --end-week 22 \
  --persist \
  --run-trigger logged-replay-2025-26-weeks-02-22
```

Generate an interview-ready Markdown report:

```bash
python -m evaluation.evaluate_logged_predictions \
  --season 2025-26 \
  --start-week 2 \
  --end-week 22 \
  --persist \
  --run-trigger logged-replay-2025-26-weeks-02-22 \
  --output reports/logged_replay_evaluation_2025_26.md
```

The logged evaluator:

- Selects dashboard runs matching `dashboard-season-2025-26-week-XX`.
- Uses the latest `prediction_runs` row per matchweek when duplicates exist.
- Filters each Week N prediction payload to `round_label == R<N>` before scoring.
- Joins stored prediction payloads to completed actual results from `rpc_wsl_weekly_stats()`.
- Reports unmatched predictions and unmatched actuals rather than silently dropping them.
- Produces Brier score, log loss, accuracy, calibration bins, confidence bucket performance, best predictions, and worst misses.

Strict round-label filtering is required because some verified replay date windows include postponed or rescheduled fixtures from other rounds. For example, a long Week 16 date window can contain R20, R21, or R14 fixtures. Logged replay evaluation therefore treats the dashboard run trigger as the intended week and excludes any stored prediction rows whose `round_label` does not match that week. Excluded rows are reported in `data_snapshot.excluded_round_mismatches` and in the Markdown report.

This is the primary evaluation method for interview evidence from the replay workflow.

## Market-Implied Benchmark

The market benchmark derives proportional no-vig 1X2 probabilities directly
from raw odds columns and evaluates them as an external market probability
reference. It is not model training data, does not create model features, and
does not change production prediction behaviour.

Run the league-only benchmark against the ignored local full CSV:

```bash
python scripts/run_market_benchmark.py \
  --csv data/exports/wsl_results_probabilities_2025_2026.csv \
  --output-md reports/market_benchmark_2025_26.md \
  --output-json reports/market_benchmark_2025_26.json \
  --output-rows reports/market_benchmark_2025_26_rows.csv
```

The evaluator excludes non-league rows with a non-empty `Note` by default,
derives actual H/D/A outcomes from goals, derives benchmark probabilities from
`Odds_1`, `Odds_X` and `Odds_2`, checks supplied implied/de-vigged probability
columns as diagnostics, and produces Markdown, JSON and row-level CSV
artefacts. Use safe language such as "market-implied benchmark" or "external
market probability reference"; keep conclusions limited by odds source,
snapshot timing and licensing.

## Model vs Market-Implied Comparison

The model-vs-market comparison is an offline matched-fixture comparison between
existing model prediction artefacts and the Phase 1 market-implied benchmark. It
does not use market odds as training data, does not create model features, does
not implement market blending and does not change production prediction
behaviour.

Run the primary comparison:

```bash
python scripts/run_model_market_comparison.py \
  --model-json reports/model_comparison_first_run.json \
  --market-csv data/exports/wsl_results_probabilities_2025_2026.csv \
  --output-md reports/model_vs_market_comparison_2025_26.md \
  --output-json reports/model_vs_market_comparison_2025_26.json \
  --output-rows reports/model_vs_market_comparison_2025_26_rows.csv
```

The runner normalizes model probabilities to unit scale, derives market
probabilities from raw odds through `evaluation.market_benchmark`, matches on
normalized date/home/away fixture keys, reports unmatched fixtures and scores
both model and market rows on the same matched fixture set.

## Unit Testing

Unit tests do not require live Supabase credentials. They pass a local DataFrame fixture directly into `run_walk_forward_evaluation()`.

Covered test areas:

- Brier score correctness.
- Log loss correctness.
- Outcome accuracy.
- Calibration bin shape.
- Confidence bucket summary shape.
- Invalid probability handling.
- Evaluation runner smoke test with a local DataFrame.

## Persistence

Evaluation persistence writes to a dedicated `evaluation_runs` table. It does not overload or change the existing `prediction_runs` table.

Create the table once with:

```bash
scripts/setup_evaluation_runs_table.sql
```

Stored fields include:

- Evaluation type and date window.
- Model config and evaluation parameters.
- Aggregate metrics.
- Calibration bins.
- Confidence buckets.
- Per-match results.
- Data snapshot metadata.
- Run trigger, optional code version, and optional notes.

Insert failures are logged and return an empty `run_id`, preserving JSON output for the manual evaluation run.

## Operational Use

Use the logged prediction evaluator after dashboard replay runs are logged. Use the walk-forward runner before model or dependency changes that could affect future predictions. Compare:

- Number of matches evaluated.
- Brier score.
- Log loss.
- Accuracy.
- Calibration by confidence bucket.

Any material metric movement should be reviewed before release.
