# Time-Decay and xG-Weighting Variants

## Purpose

Phase 8A-4 tests whether small champion-family changes to recency weighting
and xG shrinkage improve evaluation metrics. The production `champion_dc_xg`
model remains unchanged and operationally authoritative.

This experiment is offline only. It uses wrapper classes around the existing
champion evaluation adapter and does not modify `model/wsl_xg_model.py`.

## Why Time-Decay Matters

WSL teams can change quickly because of transfers, injuries, managerial
changes, fixture congestion, and uneven early-season samples. A shorter
half-life can react faster to recent information, while a longer half-life can
reduce noise by keeping older matches influential for longer.

The current reference half-life is 60 days. This phase tests only nearby
alternatives rather than a broad search.

## Why xG Weighting and Shrinkage Matter

The champion family estimates team strengths from non-penalty xG. xG is less
noisy than goals, but small WSL samples can still produce unstable team effects.
Ridge strength, xG pseudocounts, and penalty-rate shrinkage all control how
aggressively the model trusts recent observed xG signals.

## Predeclared Grid

The grid is deliberately small:

| Variant | Change |
| --- | --- |
| `champion_dc_xg` | Frozen champion reference configuration. |
| `dc_fit_rho_each_fold` | Best Phase 8A-3 comparison reference; fits rho inside each fold. |
| `txg_decay_45d` | Slightly shorter 45-day recency half-life. |
| `txg_decay_90d` | Slightly longer 90-day recency half-life. |
| `txg_alpha_025` | Moderately stronger xG team-strength ridge shrinkage. |
| `txg_xg_pseudocount_010` | Conservative xG pseudocount increase. |
| `txg_conservative_weighting` | Combined conservative alpha, xG pseudocount, and penalty-rate shrinkage. |

Raw-goal/xG blending is not included in this phase because it is not cleanly
supported by the existing champion adapter without changing production model
internals.

## Running The Experiment

```powershell
.\.venv\Scripts\python.exe scripts\run_time_decay_xg_variant_experiment.py --csv data\exports\wsl_match_data.csv --test-start 2025-10-01 --test-end 2026-05-16 --min-train-matches 12 --output-md reports\time_decay_xg_variants_first_run.md --output-json reports\time_decay_xg_variants_first_run.json
```

The runner reads a local CSV and does not require live Supabase access.

## Interpreting Results

No production promotion should happen from this PR alone. Any improvement over
`champion_dc_xg` or `dc_fit_rho_each_fold` should be treated as a signal for
follow-up validation across more seasons or future WSL matchweeks, not as proof
that the operational model should change immediately.
