# Champion Calibration

Phase 7A tests whether the current champion probabilities can be improved by a post-hoc calibration layer. This is an offline evaluation experiment only. It does not change the Dixon-Coles/xG champion implementation, API behaviour, dashboard behaviour, or production prediction outputs.

The first implementation uses temperature scaling plus shrinkage toward base outcome rates. The calibrator is fit on earlier champion folds and evaluated on later champion folds from the same generated model-comparison artefact.

## Why Calibration Comes After Diagnostics

The model comparison established `champion_dc_xg` as the reference model. Diagnostics then highlighted where confidence and errors deserve attention. Calibration is the next small test because it can improve probability quality without introducing a new model family or changing the champion's fixture logic.

## How To Run

```powershell
.\.venv\Scripts\python.exe scripts\run_champion_calibration_experiment.py `
  --input-json reports\model_comparison_first_run.json `
  --output-md reports\champion_calibration_first_run.md `
  --output-json reports\champion_calibration_first_run.json
```

The script reads local JSON only and requires no live Supabase access.

## How To Interpret Results

The report compares original champion probabilities against calibrated probabilities on the later trial folds. Brier score and log loss are the main probability-quality metrics. Accuracy is still reported, but calibration can be useful even when accuracy is flat or slightly worse, because calibration changes probability sharpness rather than the underlying model's ranking logic.

If calibrated probabilities improve log loss or Brier score on later folds, they should be considered for future production-style testing. If they do not improve these metrics, keep the uncalibrated champion as the reference and continue offline experiments only.

## Guardrails

- Fit calibration parameters only on earlier folds.
- Evaluate only on later folds for the reported experiment.
- Treat results from 109 evaluated champion matches as a hypothesis, not a final production decision.
- Do not replace `champion_dc_xg` until calibrated outputs win repeated backtests and the production integration is explicitly requested.
