# Champion Diagnostics

Champion diagnostics comes after model comparison because comparison answers which model is strongest on shared folds, while diagnostics asks why that model wins, where it fails, and what should be tested next.

The first diagnostics target is `champion_dc_xg`, using the generated model-comparison artefact from the first real comparison run. The diagnostics script reads local JSON rows only; it does not fit models, call Supabase, change prediction outputs, or alter API/dashboard behaviour.

## Questions Answered

- Which high-confidence champion predictions were wrong?
- Which high-confidence champion predictions were strongest?
- Does the champion behave differently for home favourites, away favourites, and predicted draws?
- Which confidence bands look strongest or weakest?
- Are errors concentrated by team, round, or fold?
- How does the champion compare with the nearest challenger on the same fixtures?

## How To Run

```powershell
.\.venv\Scripts\python.exe scripts\run_champion_diagnostics.py `
  --input-json reports\model_comparison_first_run.json `
  --output-md reports\champion_diagnostics_first_run.md `
  --output-json reports\champion_diagnostics_first_run.json
```

Use `--top-n` to change the number of rows in ranked tables and `--high-confidence` to change the cutoff for high-confidence misses and correct predictions.

## How To Interpret Outputs

The Markdown report is the primary human-readable output. It starts with champion headline metrics, then compares the champion against the ranked model comparison table. Diagnostic sections highlight high-confidence misses, high-confidence correct predictions, confidence-band performance, favourite/draw behaviour, team-level error summaries, round/fold instability, and paired champion-versus-challenger results.

The optional JSON output contains the same summary in structured form for future analysis or dashboarding.

These reports should be read as diagnostics for one generated artefact, not as proof of a permanent modelling truth. The first run covers one season slice and 109 champion-evaluated matches.

## How Diagnostics Informs The Next Phase

Diagnostics should shape the next challenger or improvement phase by pointing to specific hypotheses. For example, overconfident misses suggest calibration work, repeated team-level residuals suggest feature or team-strength review, and weak draw behaviour suggests draw-specific calibration or features.

The next modelling PR should test one or two of those hypotheses on the same rolling folds and compare against `champion_dc_xg` without changing the champion reference implementation.
