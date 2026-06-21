# Regularised Team Strength Challenger

`regularised_team_strength` is Phase 8A-1 of the reconciled model roadmap. It adds a conservative statistical challenger without changing `champion_dc_xg`, API behaviour, dashboard behaviour, or production prediction outputs.

## Why This Model Exists

The first comparison showed `champion_dc_xg` remains strongest, but only the naive, Elo, and logistic challengers have been tested. This model fills the next gap: a regularised attack/defence Poisson challenger that is simpler than the champion but more football-specific than Elo or logistic regression.

## How It Works

The model fits on training-fold rows only. It prefers non-penalty xG columns (`home_np_xg`, `away_np_xg`), falls back to xG (`home_xg`, `away_xg`), and finally falls back to goals if xG is unavailable.

For each team it estimates:

- attacking strength: smoothed attacking xG/goals for divided by league average
- defensive weakness: smoothed xG/goals conceded divided by league average

The smoothing parameter `shrinkage_matches` acts like pseudo-matches at league-average strength. Teams with little data are pulled toward 1.0, and unseen teams use league-average attack and defence.

Fixture expected goals are:

```text
home_expected = league_home_rate * home_attack * away_defence_weakness
away_expected = league_away_rate * away_attack * home_defence_weakness
```

Expected goals are clipped between `min_rate` and `max_rate`, then converted into home/draw/away probabilities using an independent Poisson scoreline grid.

## How It Differs

- Compared with `elo_baseline`, it uses attacking and defensive production instead of result ratings.
- Compared with `logistic_regression`, it is a transparent generative score model rather than a feature classifier.
- Compared with `champion_dc_xg`, it is simpler: no Dixon-Coles correction, no penalty component, no ridge log-linear xG fit, and no production prediction path.

## How To Run

```powershell
.\.venv\Scripts\python.exe scripts\run_model_comparison.py `
  --csv data\exports\wsl_match_data.csv `
  --model champion_dc_xg `
  --model regularised_team_strength `
  --model logistic_regression `
  --model elo_baseline `
  --model naive_outcome_rate `
  --test-start 2025-10-01 `
  --test-end 2026-05-16 `
  --min-train-matches 12 `
  --output-md reports\regularised_team_strength_comparison_first_run.md `
  --output-json reports\regularised_team_strength_comparison_first_run.json
```

## How To Interpret Outputs

Brier score and log loss are the primary probability-quality metrics. Accuracy is useful but secondary. This model should be considered useful only if it beats the existing challengers on identical rolling folds and closes the gap to `champion_dc_xg`.

## First Run Result

The first local comparison over 109 evaluated matches ranked:

| rank | model | Brier | log loss | accuracy |
| --- | --- | --- | --- | --- |
| 1 | `champion_dc_xg` | 0.5052 | 0.8669 | 0.6239 |
| 2 | `regularised_team_strength` | 0.5256 | 0.9001 | 0.5963 |
| 3 | `logistic_regression` | 0.5562 | 0.9560 | 0.5780 |
| 4 | `elo_baseline` | 0.5676 | 0.9596 | 0.5596 |
| 5 | `naive_outcome_rate` | 0.6401 | 1.0576 | 0.4587 |

The regularised team-strength model becomes the strongest tested challenger, but it does not beat the champion. The champion remains the reference model.

## Limitations

- One WSL season and 109 evaluated matches are not enough for a final model-selection decision.
- The model assumes independent home and away Poisson scoring.
- It does not model low-score correlation, penalty components, player availability, or market information.
- Shrinkage improves robustness but can understate genuinely dominant teams.
