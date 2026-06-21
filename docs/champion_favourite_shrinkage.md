# Champion Favourite Shrinkage

Phase 7C tests whether the current champion is too confident on dominant teams and other strong favourites. This is an offline experiment only. It does not change `model/wsl_xg_model.py`, production prediction outputs, API behaviour, or dashboard behaviour.

## Why This Comes After Draw Adjustment

Phase 7A tested broad calibration and Phase 7B tested draw-specific adjustment. Favourite shrinkage is narrower again: it asks whether extreme favourite probabilities should be softened while preserving the champion model as the reference.

## How To Run

```powershell
.\.venv\Scripts\python.exe scripts\run_champion_favourite_shrinkage_experiment.py `
  --input-json reports\model_comparison_first_run.json `
  --output-md reports\champion_favourite_shrinkage_first_run.md `
  --output-json reports\champion_favourite_shrinkage_first_run.json
```

The script reads local JSON only and requires no live Supabase access.

## Variants Tested

The first experiment compares the original champion against:

- Threshold shrinkage for favourites above `0.65`, `0.70`, `0.75`, and `0.80`
- Shrink strengths of `0.05`, `0.10`, and `0.15`
- Soft-cap shrinkage with caps of `0.70`, `0.75`, and `0.80`
- Soft-cap strengths of `0.25`, `0.50`, and `0.75`

Removed favourite probability mass is redistributed proportionally across the other two outcomes. Rows below the threshold or cap are unchanged.

## How To Interpret Results

Brier score and log loss are the main probability-quality metrics. Accuracy is still shown because changing the top class matters, but favourite shrinkage can still be worth shadow testing if it improves probability quality while reducing accuracy.

High-confidence favourite metrics show how many strong favourite calls remain, how accurate they are, how many become misses or correct calls, and the log loss on high-confidence misses. If shrinkage reduces misses but damages too many high-confidence correct calls, keep it offline.

Keep the original champion unchanged unless a shrinkage variant wins repeated offline tests and is explicitly promoted in a separate production-focused change.
