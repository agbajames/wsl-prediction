# Evaluation Strategy

Last updated: 2026-06-14

The evaluation baseline makes the WSL Prediction Engine measurable without changing the core model logic in `model/wsl_xg_model.py`.

## Components

- `evaluation/metrics.py`: reusable three-way football prediction metrics.
- `evaluation/run_evaluation.py`: repeatable walk-forward evaluation runner.
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

Manual evaluation uses the existing Supabase data layer:

```bash
python -m evaluation.run_evaluation --start-date 2025-10-01
```

The runner:

- Fetches match data through `data.supabase_client.fetch_match_data()` when no DataFrame is provided.
- Supports local or mocked DataFrames in tests.
- Calls the existing model `run_backtest()` function.
- Returns a structured JSON-style result with parameters, aggregate metrics, model backtest metrics, and per-match results.

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

Dedicated evaluation-run persistence is deferred. The existing `prediction_runs` audit table remains unchanged.

Recommended next persistence step:

- Add `evaluation_runs` as a separate table rather than overloading `prediction_runs`.
- Store evaluation parameters, aggregate metrics, confidence buckets, calibration bins, per-match results, code/image version, and run trigger.
- Keep prediction-run logging backward compatible.

## Operational Use

Use this evaluation runner before model or dependency changes that could affect predictions. Compare:

- Number of matches evaluated.
- Brier score.
- Log loss.
- Accuracy.
- Calibration by confidence bucket.

Any material metric movement should be reviewed before release.
