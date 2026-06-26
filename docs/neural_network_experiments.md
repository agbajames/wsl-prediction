# Neural Network Experiments

## What Was Implemented

Phase 2 adds a rigorous, dependency-light neural-network experiment harness while keeping `neural_network` as a challenger model only. The champion implementation in `models/champion_dc_xg.py` and `model/wsl_xg_model.py` is not modified.

`models/neural_network.py` now supports a small NumPy architecture ladder:

| Stage | Hidden layers | Purpose |
| --- | --- | --- |
| NN-0 | `[]` | No-hidden-layer softmax regression baseline over the same scaled features. |
| NN-1 | `[8]` | Tiny MLP, matching the spirit of the original proof of concept. |
| NN-1b | `[16]` | Slightly wider tiny MLP. |
| NN-2 | `[32]` | Moderate one-layer capacity test. |
| NN-2b | `[64]` | Wider one-layer capacity test. |
| NN-3 | `[32, 16]` | Small two-layer MLP. |
| NN-3b | `[64, 32]` | Wider two-layer MLP. |

The existing registry entry remains `neural_network`. Existing callers that pass `hidden_units=8` still get a one-hidden-layer model. New experiment callers can pass `hidden_layers=()` for NN-0 or tuples such as `(32, 16)`.

## How To Run

Use the dedicated runner:

```bash
python scripts/run_neural_network_experiments.py \
  --csv data/exports/wsl_match_data.csv \
  --test-start 2025-10-01 \
  --test-end 2026-05-16 \
  --min-train-matches 12
```

The default run uses the `xg` feature group, the seven predeclared architectures, and seeds `42`, `7`, and `123`. It also attempts to include the existing baseline/challenger models where the supplied CSV has the columns they require.

The runner now protects the xG experiment from accidental fallback-to-goals. When the predeclared `xg` experiments are run, the CSV must contain `home_np_xg`, `away_np_xg`, `home_xg`, and `away_xg`, with usable xG values for completed rows. If those columns are absent or incomplete, the run fails before model fitting.

Useful options:

```bash
python scripts/run_neural_network_experiments.py \
  --csv data/exports/wsl_match_data.csv \
  --test-start 2025-10-01 \
  --test-end 2026-05-16 \
  --seed 42 --seed 7 --seed 123 \
  --l2-penalty 0.001 \
  --dropout 0 \
  --validation-fraction 0.2 \
  --early-stopping-patience 20
```

Only for an explicit results-only diagnostic, add:

```bash
python scripts/run_neural_network_experiments.py \
  --csv data/exports/wsl_results_only.csv \
  --test-start 2025-10-01 \
  --test-end 2026-05-16 \
  --allow-xg-fallback-to-goals
```

Do not use that flag for a claimed xG experiment.

This is deliberately not a large grid search. If you want to test L2 values such as `0`, `1e-5`, `1e-4`, `1e-3`, or `1e-2`, run them as separate controlled experiments and keep the final comparison window fixed.

## Early Stopping

Early stopping uses a time-aware validation split inside each training fold:

- Training rows are already ordered by match date.
- The latest portion of each training fold becomes validation data.
- Earlier rows remain the fitting subset.
- The scaler is fit on the fitting subset only.
- Validation rows are never randomly sampled.
- The best validation-loss weights are restored before prediction.

If a fold is too small for a useful validation split, early stopping is skipped for that fold and the model records the skip reason in `fit_diagnostics`.

## Outputs

The runner writes:

| File | Contents |
| --- | --- |
| `reports/neural_network_experiments_summary.csv` | Mean/std metrics by experiment, best/worst seed, seed-stability flag. |
| `reports/neural_network_experiments_seed_metrics.csv` | One aggregate metric row per experiment and seed. |
| `reports/neural_network_experiments_fold_metrics.csv` | Fold-level metrics for each experiment and seed. |
| `reports/neural_network_experiments_predictions.csv` | Row-level predictions and actual outcomes. |
| `reports/neural_network_experiments_training_history.csv` | Train/validation loss per epoch/fold for NN runs. |
| `reports/neural_network_experiments_metadata.json` | Backtest config, folds, and warnings. |
| `reports/neural_network_experiments.md` | Markdown summary ranked by log loss, then Brier score, then accuracy. |

Diagnostics include mean predicted probability by class, average max confidence, high-confidence wrong counts, actual-outcome probability below 5% and 10%, confusion matrix cells, and per-class Brier scores.

## Interpreting Results

Prioritise probability quality:

1. Log loss.
2. Brier score.
3. Accuracy.

Accuracy is not enough. A neural network that improves accuracy while worsening log loss or Brier score is probably too sharp or poorly calibrated for this use case.

Train vs validation divergence matters:

- Falling train loss with rising validation loss suggests overfitting.
- Large seed-to-seed standard deviation suggests instability.
- Many high-confidence wrong predictions or actual probabilities below 5% are overconfidence warnings.
- NN-0 should behave like a logistic sanity baseline. If it is erratic, inspect preprocessing before trusting deeper models.

## Continue Or Stop Criteria

Continue neural-network work only if small models improve or nearly match the existing ML challengers on log loss and Brier score across folds and seeds, while keeping overconfidence under control.

Stop or keep the work exploratory if:

- Larger architectures only improve accuracy.
- Results depend on one seed.
- Validation loss diverges from training loss.
- The NN cannot beat or explainably complement `improved_logistic_regression`, `random_forest`, or `regularised_team_strength`.

The champion model remains the reference until a challenger beats it fairly on time-aware probability-quality metrics and preferably shadow/live-style evaluations.
