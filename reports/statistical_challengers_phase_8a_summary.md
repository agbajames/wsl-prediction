# Phase 8A Consolidated Statistical Challengers Report

## Scope

This report consolidates the Phase 8A first-run results for standalone statistical challengers,
Dixon-Coles/champion-family configuration variants, and time-decay/xG-weighting variants.
It is documentation/reporting only and does not change production prediction behaviour.

All headline rows use the same local 2025-10-01 to 2026-05-16 comparison window where available,
with 109 evaluated matches. Lower Brier score and log loss are better; higher accuracy is better.

## Primary Model Summary

| model_name | n_matches | brier_score | log_loss | accuracy | note |
| --- | --- | --- | --- | --- | --- |
| dc_fit_rho_each_fold | 109 | 0.5027 | 0.8623 | 0.6147 | Best Phase 8A probability-quality candidate; fits rho inside each fold. |
| txg_xg_pseudocount_010 | 109 | 0.5038 | 0.8649 | 0.6330 | Most balanced champion-family candidate; improves Brier, log loss, and accuracy versus the original champion. |
| champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 | Unchanged operational/reference model. |
| regularised_team_strength | 109 | 0.5256 | 0.9001 | 0.5963 | Strongest standalone non-champion statistical challenger. |
| poisson_regression | 109 | 0.5585 | 0.9426 | 0.5413 | Interpretable Poisson regression challenger; useful but below regularised team strength. |
| logistic_regression | 109 | 0.5562 | 0.9560 | 0.5780 | Existing feature-based ML baseline carried forward to Phase 8B. |
| elo_baseline | 109 | 0.5676 | 0.9596 | 0.5596 | Simple rating baseline. |
| naive_outcome_rate | 109 | 0.6401 | 1.0576 | 0.4587 | Sanity-check baseline. |

## Ranking By Brier Score

| rank | model_name | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- |
| 1 | dc_fit_rho_each_fold | 0.5027 | 0.8623 | 0.6147 |
| 2 | txg_xg_pseudocount_010 | 0.5038 | 0.8649 | 0.6330 |
| 3 | champion_dc_xg | 0.5052 | 0.8669 | 0.6239 |
| 4 | regularised_team_strength | 0.5256 | 0.9001 | 0.5963 |
| 5 | logistic_regression | 0.5562 | 0.9560 | 0.5780 |
| 6 | poisson_regression | 0.5585 | 0.9426 | 0.5413 |
| 7 | elo_baseline | 0.5676 | 0.9596 | 0.5596 |
| 8 | naive_outcome_rate | 0.6401 | 1.0576 | 0.4587 |

## Ranking By Log Loss

| rank | model_name | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- |
| 1 | dc_fit_rho_each_fold | 0.5027 | 0.8623 | 0.6147 |
| 2 | txg_xg_pseudocount_010 | 0.5038 | 0.8649 | 0.6330 |
| 3 | champion_dc_xg | 0.5052 | 0.8669 | 0.6239 |
| 4 | regularised_team_strength | 0.5256 | 0.9001 | 0.5963 |
| 5 | poisson_regression | 0.5585 | 0.9426 | 0.5413 |
| 6 | logistic_regression | 0.5562 | 0.9560 | 0.5780 |
| 7 | elo_baseline | 0.5676 | 0.9596 | 0.5596 |
| 8 | naive_outcome_rate | 0.6401 | 1.0576 | 0.4587 |

## Ranking By Accuracy

| rank | model_name | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- |
| 1 | txg_xg_pseudocount_010 | 0.5038 | 0.8649 | 0.6330 |
| 2 | champion_dc_xg | 0.5052 | 0.8669 | 0.6239 |
| 3 | dc_fit_rho_each_fold | 0.5027 | 0.8623 | 0.6147 |
| 4 | regularised_team_strength | 0.5256 | 0.9001 | 0.5963 |
| 5 | logistic_regression | 0.5562 | 0.9560 | 0.5780 |
| 6 | elo_baseline | 0.5676 | 0.9596 | 0.5596 |
| 7 | poisson_regression | 0.5585 | 0.9426 | 0.5413 |
| 8 | naive_outcome_rate | 0.6401 | 1.0576 | 0.4587 |

## Supporting Phase 8A Variants

These rows are useful context, but they are not the main carry-forward candidates.

| model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- |
| dc_rho_mild_minus_08 | 109 | 0.5041 | 0.8650 | 0.6330 |
| dc_conservative_xg_shrinkage | 109 | 0.5052 | 0.8669 | 0.6239 |
| dc_score_grid_10 | 109 | 0.5052 | 0.8670 | 0.6239 |
| dc_alpha_030 | 109 | 0.5058 | 0.8681 | 0.6147 |
| dc_rho_stronger_minus_18 | 109 | 0.5064 | 0.8691 | 0.6147 |
| txg_decay_90d | 109 | 0.5055 | 0.8672 | 0.6147 |
| txg_alpha_025 | 109 | 0.5056 | 0.8677 | 0.6239 |
| txg_conservative_weighting | 109 | 0.5058 | 0.8682 | 0.6239 |
| txg_decay_45d | 109 | 0.5061 | 0.8684 | 0.6239 |

## Interpretation

- Best probability-quality candidate: `dc_fit_rho_each_fold` with Brier 0.5027 and log loss 0.8623.
- Best accuracy candidate among primary rows: `txg_xg_pseudocount_010` with accuracy 0.6330.
- Strongest standalone non-champion challenger: `regularised_team_strength` with Brier 0.5256, log loss 0.9001, and accuracy 0.5963.
- `champion_dc_xg` remains the operational/reference model because the improvements are from one
  evaluation window, some candidates trade probability quality against accuracy, and no candidate has
  passed shadow/live-style validation yet.

## Model Decision

- No production promotion yet.
- Carry `dc_fit_rho_each_fold` forward as the current best probability-quality candidate.
- Carry `txg_xg_pseudocount_010` forward as the most balanced champion-family candidate because it
  improves Brier score, log loss, and accuracy versus the original champion.
- Keep `regularised_team_strength` as the strongest standalone non-champion statistical challenger.
- Phase 8A is complete.
- Next phase: Phase 8B - improved logistic regression and feature-based ML challengers.

## Limitations

- The main comparison covers one WSL season and 109 evaluated matches.
- The WSL sample is small, so small differences can be unstable.
- Champion-family variants were deliberately predeclared, but repeated testing still risks overfitting
  to this evaluation window.
- Candidate variants need shadow/live-style validation before any production decision.
- Accuracy is a coarse metric and can disagree with probability-quality metrics such as Brier score and log loss.

## Source Artifacts

- `standalone_and_baselines`: `reports/poisson_regression_comparison_first_run.json`
- `dixon_coles_variants`: `reports/dixon_coles_variants_first_run.json`
- `time_decay_xg_variants`: `reports/time_decay_xg_variants_first_run.json`
