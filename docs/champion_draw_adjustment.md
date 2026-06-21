# Champion Draw Adjustment

Phase 7B tests whether the current champion's 1X2 probabilities benefit from a simple post-processing adjustment to draw probability. This is an offline experiment only. It does not change `model/wsl_xg_model.py`, production prediction outputs, API behaviour, or dashboard behaviour.

## Why This Comes After Calibration

Phase 7A tested a broader calibration layer and did not improve the first-run trial metrics. Draw adjustment is narrower: it asks whether the champion specifically under-prices or over-prices draws, using fixed, interpretable adjustments instead of fitted optimisation.

## How To Run

```powershell
.\.venv\Scripts\python.exe scripts\run_champion_draw_adjustment_experiment.py `
  --input-json reports\model_comparison_first_run.json `
  --output-md reports\champion_draw_adjustment_first_run.md `
  --output-json reports\champion_draw_adjustment_first_run.json
```

The script reads local JSON only and requires no live Supabase access.

## Variants Tested

The first experiment compares the original champion against:

- Additive draw adjustments: `-0.05`, `-0.025`, `+0.025`, `+0.05`
- Multiplicative draw adjustments: `0.85`, `0.95`, `1.05`, `1.15`

Additive uplift adds mass to `p_draw` and removes it proportionally from `p_home_win` and `p_away_win`. Additive shrinkage does the reverse. Multiplicative variants scale `p_draw` and then renormalize all three probabilities.

## How To Interpret Results

Brier score and log loss are the main probability-quality metrics. Accuracy is reported because top-class decisions matter, but a variant can still be interesting if it improves Brier or log loss while reducing accuracy.

Draw-specific metrics show actual draw count, draw prediction rate, average predicted draw probability, draw recall, draw log loss, and non-draw performance. If draw recall improves while overall Brier/log loss worsens, that is a warning rather than a production signal.

Keep the original champion unchanged unless a draw-adjusted variant wins repeated offline tests on probability quality and is explicitly promoted in a separate production-focused change.
