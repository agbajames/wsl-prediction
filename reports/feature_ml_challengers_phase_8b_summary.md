# Phase 8B Consolidated Feature-Based ML Challengers Report

## Scope

This report consolidates Phase 8B feature-based ML challenger results and compares them
with the Phase 8A statistical/champion-family leaders. It is documentation/reporting only
and does not change production prediction behaviour.

All headline rows use the same local 2025-10-01 to 2026-05-16 comparison window where available,
with 109 evaluated matches. Lower Brier score and log loss are better; higher accuracy is better.

## Primary Model Summary

| model_name | n_matches | brier_score | log_loss | accuracy | note |
| --- | --- | --- | --- | --- | --- |
| dc_fit_rho_each_fold | 109 | 0.5027 | 0.8623 | 0.6147 | Phase 8A best probability-quality candidate; not promoted. |
| txg_xg_pseudocount_010 | 109 | 0.5038 | 0.8649 | 0.6330 | Phase 8A most balanced champion-family candidate; not promoted. |
| champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 | Unchanged operational/reference model. |
| regularised_team_strength | 109 | 0.5256 | 0.9001 | 0.5963 | Strongest standalone statistical challenger. |
| improved_logistic_regression | 109 | 0.5384 | 0.9356 | 0.5780 | Best feature-based ML challenger on Brier score and log loss. |
| random_forest | 109 | 0.5491 | 0.9385 | 0.5872 | Conservative tree-based challenger; best feature-ML accuracy in Phase 8B. |
| logistic_regression | 109 | 0.5562 | 0.9560 | 0.5780 | Original feature-based ML baseline. |
| poisson_regression | 109 | 0.5585 | 0.9426 | 0.5413 | Interpretable statistical challenger. |
| elo_baseline | 109 | 0.5676 | 0.9596 | 0.5596 | Simple rating baseline. |
| naive_outcome_rate | 109 | 0.6401 | 1.0576 | 0.4587 | Sanity-check baseline. |

## Ranking By Brier Score

| rank | model_name | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- |
| 1 | dc_fit_rho_each_fold | 0.5027 | 0.8623 | 0.6147 |
| 2 | txg_xg_pseudocount_010 | 0.5038 | 0.8649 | 0.6330 |
| 3 | champion_dc_xg | 0.5052 | 0.8669 | 0.6239 |
| 4 | regularised_team_strength | 0.5256 | 0.9001 | 0.5963 |
| 5 | improved_logistic_regression | 0.5384 | 0.9356 | 0.5780 |
| 6 | random_forest | 0.5491 | 0.9385 | 0.5872 |
| 7 | logistic_regression | 0.5562 | 0.9560 | 0.5780 |
| 8 | poisson_regression | 0.5585 | 0.9426 | 0.5413 |
| 9 | elo_baseline | 0.5676 | 0.9596 | 0.5596 |
| 10 | naive_outcome_rate | 0.6401 | 1.0576 | 0.4587 |

## Ranking By Log Loss

| rank | model_name | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- |
| 1 | dc_fit_rho_each_fold | 0.5027 | 0.8623 | 0.6147 |
| 2 | txg_xg_pseudocount_010 | 0.5038 | 0.8649 | 0.6330 |
| 3 | champion_dc_xg | 0.5052 | 0.8669 | 0.6239 |
| 4 | regularised_team_strength | 0.5256 | 0.9001 | 0.5963 |
| 5 | improved_logistic_regression | 0.5384 | 0.9356 | 0.5780 |
| 6 | random_forest | 0.5491 | 0.9385 | 0.5872 |
| 7 | poisson_regression | 0.5585 | 0.9426 | 0.5413 |
| 8 | logistic_regression | 0.5562 | 0.9560 | 0.5780 |
| 9 | elo_baseline | 0.5676 | 0.9596 | 0.5596 |
| 10 | naive_outcome_rate | 0.6401 | 1.0576 | 0.4587 |

## Ranking By Accuracy

| rank | model_name | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- |
| 1 | txg_xg_pseudocount_010 | 0.5038 | 0.8649 | 0.6330 |
| 2 | champion_dc_xg | 0.5052 | 0.8669 | 0.6239 |
| 3 | dc_fit_rho_each_fold | 0.5027 | 0.8623 | 0.6147 |
| 4 | regularised_team_strength | 0.5256 | 0.9001 | 0.5963 |
| 5 | random_forest | 0.5491 | 0.9385 | 0.5872 |
| 6 | improved_logistic_regression | 0.5384 | 0.9356 | 0.5780 |
| 7 | logistic_regression | 0.5562 | 0.9560 | 0.5780 |
| 8 | elo_baseline | 0.5676 | 0.9596 | 0.5596 |
| 9 | poisson_regression | 0.5585 | 0.9426 | 0.5413 |
| 10 | naive_outcome_rate | 0.6401 | 1.0576 | 0.4587 |

## Phase 8B Interpretation

- Best feature-based ML model by probability quality: `improved_logistic_regression` with Brier 0.5384 and log loss 0.9356.
- Best feature-based ML model by accuracy: `random_forest` with accuracy 0.5872.
- Tree-based modelling added value over the original logistic baseline on log loss and accuracy,
  but it did not beat improved logistic regression on Brier score or log loss.
- Feature-based ML improved the ML baselines, but still trails the statistical/champion-family leaders.

## Feature And Ablation Notes

- The improved logistic `xg` feature group was the best ablation and became the registered default.
- The full improved-logistic feature set underperformed, which is a useful warning about feature richness
  on a one-season WSL sample.
- The random-forest feature-importance artifact is available at
  `reports/tree_based_challenger_feature_importance_first_run.md`.

### Improved Logistic Ablation

| model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- |
| improved_logistic_xg | 109 | 0.5384 | 0.9356 | 0.5780 |
| improved_logistic_base | 109 | 0.5529 | 0.9496 | 0.5688 |
| improved_logistic_opponent | 109 | 0.5530 | 0.9521 | 0.5688 |
| improved_logistic_full | 109 | 0.5786 | 1.0112 | 0.5505 |
| improved_logistic_form | 109 | 0.5878 | 1.0145 | 0.5321 |

## Model Decision

- No production promotion yet.
- Keep `champion_dc_xg` as the operational/reference model.
- Carry `dc_fit_rho_each_fold` forward as the best overall probability-quality candidate.
- Carry `txg_xg_pseudocount_010` forward as the most balanced champion-family candidate.
- Carry `regularised_team_strength` forward as the strongest standalone statistical challenger.
- Carry `improved_logistic_regression` and `random_forest` forward as Phase 8B ML benchmarks.

## Next-Step Recommendation

Recommendation: prioritise Phase 8D ensemble/blending and Phase 8E shadow testing before any
production model decision. Phase 8C neural-network work should be parked or kept explicitly
research-only because one WSL season is too small for a high-capacity model to earn promotion.

## Limitations

- The main comparison covers one WSL season and 109 evaluated matches.
- The WSL dataset is small, so small metric differences can be unstable.
- Feature-rich models remain vulnerable to overfitting even with leakage-safe feature generation.
- Candidate models need shadow/live-style validation before any production decision.

## Source Artifacts

- `phase_8a_summary`: `reports/statistical_challengers_phase_8a_summary.json`
- `tree_based_comparison`: `reports/tree_based_challenger_first_run.json`
- `improved_logistic_ablation`: `reports/improved_logistic_regression_ablation_first_run.json`
- `tree_feature_importance`: `reports/tree_based_challenger_feature_importance_first_run.md`
