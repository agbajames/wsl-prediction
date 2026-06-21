# Dixon-Coles Configuration Variants

## Purpose

Phase 8A-3 adds offline champion-family configuration experiments. The
production `champion_dc_xg` model remains the reference model, and no API,
dashboard, or production prediction behaviour changes.

The goal is to test whether small, predeclared adjustments around the current
xG Dixon-Coles family improve Brier score, log loss, accuracy, or calibration
enough to justify deeper investigation.

## Variant Grid

The grid is deliberately small to avoid overfitting one WSL season.

| Variant | Change |
| --- | --- |
| `champion_dc_xg` | Frozen champion reference configuration. |
| `dc_rho_mild_minus_08` | Less negative fixed Dixon-Coles rho, `-0.08`. |
| `dc_rho_stronger_minus_18` | More negative fixed Dixon-Coles rho, `-0.18`. |
| `dc_fit_rho_each_fold` | Fit rho inside each training fold using the existing champion grid search. |
| `dc_score_grid_10` | Extend scoreline truncation from 8 to 10 goals. |
| `dc_alpha_030` | Increase xG strength ridge regularisation from `0.15` to `0.30`. |
| `dc_decay_30d` | Use a shorter 30-day time-decay half-life. |
| `dc_conservative_xg_shrinkage` | Increase np_xG pseudocount and penalty-rate shrinkage. |

## Why This Is Offline Only

These variants are wrappers around the existing champion evaluation adapter.
They use the same fitting and prediction functions as `champion_dc_xg`, but with
named local experiment configs. They are not registered as production models and
do not alter `model/wsl_xg_model.py`.

## Running The Experiment

```powershell
.\.venv\Scripts\python.exe scripts\run_dixon_coles_variant_experiment.py --csv data\exports\wsl_match_data.csv --test-start 2025-10-01 --test-end 2026-05-16 --min-train-matches 12 --output-md reports\dixon_coles_variants_first_run.md --output-json reports\dixon_coles_variants_first_run.json
```

The runner uses the same rolling-fold evaluation framework as the broader model
comparison scripts and does not require live Supabase access.

## Interpreting Results

`champion_dc_xg` remains the reference unless a variant clearly improves the
main evaluation metrics and calibration. Small metric differences over one WSL
season should be treated as directional, not definitive.
