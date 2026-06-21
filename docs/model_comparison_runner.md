# Model Comparison Runner

`scripts/run_model_comparison.py` runs the frozen champion and challenger
models on identical rolling folds, then writes a Markdown evaluation report and
optional JSON summary.

## Example

```powershell
.\.venv\Scripts\python.exe scripts\run_model_comparison.py `
  --csv data\local_matches.csv `
  --model naive_outcome_rate `
  --model elo_baseline `
  --test-start 2025-10-01 `
  --test-end 2025-11-30 `
  --min-train-matches 10 `
  --output-md reports\model_comparison.md `
  --output-json reports\model_comparison.json
```

If no `--model` flags are supplied, the runner defaults to:

- `champion_dc_xg`
- `naive_outcome_rate`
- `elo_baseline`
- `logistic_regression`

## Identical Folds

Models must use identical folds so metric differences reflect model behaviour,
not different train/test splits. The runner builds rolling folds once, then
passes the same fold objects to every model.

## Input CSV

All models need:

- `match_date`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`

The champion also needs its xG production schema:

- `round_label`
- `home_xg`
- `away_xg`
- `home_np_xg`
- `away_np_xg`

If the local CSV does not include xG columns, run baseline/logistic comparisons
only.

## Outputs

The Markdown report includes model comparison metrics, calibration bins,
confidence buckets, worst misses, best high-confidence correct predictions, and
favourite/confidence breakdowns. The optional JSON summary includes the fold
metadata, model result payloads, prediction rows, and report summary.

## Limitations

WSL samples are small, especially early in a season. Some folds may be skipped
if they do not meet `min_train_matches`, and fold-level results can be noisy.
Rescheduled fixtures are handled by match date, so round labels are metadata,
not the primary split key.

This runner supports pre-season model selection by making the champion and all
challengers compete on the same historical evidence before a new model is
promoted.

