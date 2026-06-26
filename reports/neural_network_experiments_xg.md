# Corrected xG Phase 3 Neural Network Experiment

Fresh Supabase export succeeded. This report uses the official xG export generated on 2026-06-26 and is not the earlier results-only fallback-goals diagnostic. The experiment did not pass `--allow-xg-fallback-to-goals`; the runner metadata records `allow_xg_fallback_to_goals = false` and `rows_that_would_fallback_to_goals = 0`.

## Data and Guardrails

- CSV used: `data/exports/wsl_match_data_xg_phase3_20260626_fresh.csv`
- Rows in fresh export: 132
- Completed rows: 132
- Date range: 2025-09-05 to 2026-05-16
- xG columns present: `home_np_xg, away_np_xg, home_xg, away_xg`
- xG missingness: home_np_xg=0, away_np_xg=0, home_xg=0, away_xg=0
- Fallback-to-goals avoided: yes; every completed row had xG for both teams.
- Champion model files were not modified by this run.

## Evaluation Setup

- Evaluation window: 2025-10-01 to 2026-05-16
- Rolling folds: 19
- Fixtures used in evaluation: 109
- Seeds: 42, 7, 123
- Architectures tested: `nn_logistic_xg` []; `nn_tiny_8_xg` [8]; `nn_tiny_16_xg` [16]; `nn_moderate_32_xg` [32]; `nn_moderate_64_xg` [64]; `nn_two_layer_32_16_xg` [32, 16]; `nn_two_layer_64_32_xg` [64, 32]
- Metric priority: log loss, then Brier score, then accuracy. Accuracy alone is not a promotion criterion.

## Ranked Summary

| experiment_name | n_runs | n_matches | mean_log_loss | mean_brier_score | mean_accuracy | std_log_loss | seed_stable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| champion_dc_xg | 1 | 109 | 0.8669 | 0.5052 | 0.6239 | 0.0000 | True |
| regularised_team_strength | 1 | 109 | 0.9001 | 0.5256 | 0.5963 | 0.0000 | True |
| improved_logistic_regression | 1 | 109 | 0.9356 | 0.5384 | 0.5780 | 0.0000 | True |
| random_forest | 1 | 109 | 0.9385 | 0.5491 | 0.5872 | 0.0000 | True |
| poisson_regression | 1 | 109 | 0.9426 | 0.5585 | 0.5413 | 0.0000 | True |
| nn_tiny_8_xg | 3 | 109 | 0.9746 | 0.5711 | 0.5474 | 0.0174 | True |
| nn_moderate_64_xg | 3 | 109 | 0.9823 | 0.5794 | 0.5688 | 0.0104 | True |
| nn_moderate_32_xg | 3 | 109 | 0.9850 | 0.5748 | 0.5872 | 0.0049 | True |
| nn_logistic_xg | 3 | 109 | 0.9914 | 0.5751 | 0.5474 | 0.0242 | False |
| nn_tiny_16_xg | 3 | 109 | 0.9984 | 0.5816 | 0.5535 | 0.0139 | True |
| nn_two_layer_64_32_xg | 3 | 109 | 1.0078 | 0.5892 | 0.5596 | 0.0076 | True |
| nn_two_layer_32_16_xg | 3 | 109 | 1.0202 | 0.5896 | 0.5566 | 0.0164 | True |
| naive_outcome_rate | 1 | 109 | 1.0576 | 0.6401 | 0.4587 | 0.0000 | True |

## Best Neural Networks

- Best NN by mean log loss: `nn_tiny_8_xg` with log loss 0.9746, Brier 0.5711, accuracy 0.5474.
- Best NN by mean Brier score: `nn_tiny_8_xg` with Brier 0.5711, log loss 0.9746, accuracy 0.5474.
- No NN beat `champion_dc_xg` on log loss or Brier score. Best NN log loss trailed champion by 0.1077; best NN Brier trailed champion by 0.0659.
- No NN beat `improved_logistic_regression` on log loss or Brier score. Best NN log loss trailed improved logistic by 0.0390; best NN Brier trailed improved logistic by 0.0327.

## Baseline Comparison

| experiment_name | mean_log_loss | mean_brier_score | mean_accuracy |
| --- | --- | --- | --- |
| champion_dc_xg | 0.8669 | 0.5052 | 0.6239 |
| improved_logistic_regression | 0.9356 | 0.5384 | 0.5780 |
| random_forest | 0.9385 | 0.5491 | 0.5872 |
| poisson_regression | 0.9426 | 0.5585 | 0.5413 |
| regularised_team_strength | 0.9001 | 0.5256 | 0.5963 |
| naive_outcome_rate | 1.0576 | 0.6401 | 0.4587 |

The champion remained the best model overall by the priority metrics. `regularised_team_strength` was the nearest baseline challenger, ahead of improved logistic regression on both log loss and Brier in this run. The neural models beat only `naive_outcome_rate` on the priority metrics.

## Stability and Diagnostics

Seed stability was generally acceptable for the neural ladder, except `nn_logistic_xg`, where log-loss standard deviation was 0.0242 and the runner flagged `seed_stable = false`. The best NN, `nn_tiny_8_xg`, had log-loss standard deviation 0.0174 and Brier standard deviation 0.0108 across the three seeds.

Fold-level stability was weaker for neural models than for the strongest baselines. Across seed-fold rows, `nn_tiny_8_xg` had fold log-loss standard deviation 0.2806, while `champion_dc_xg` was 0.2234 and `regularised_team_strength` was 0.1578. Small test folds make this noisy, but the neural ladder showed more fold volatility than the champion.

Training and validation loss behaviour showed mild overfit pressure as models widened or deepened. Final validation loss stayed above final training loss across the ladder; the wider and two-layer models did not translate extra capacity into better held-out log loss. The [8] model was the best calibrated NN by both priority metrics, while [32], [64], [32, 16], and [64, 32] were worse on log loss and/or Brier.

Training loss summary:

| experiment_name | mean_final_train_loss | mean_final_validation_loss | mean_best_validation_loss | median_epochs_run | mean_validation_minus_train |
| --- | --- | --- | --- | --- | --- |
| nn_moderate_32_xg | 0.7335 | 0.9147 | 0.9080 | 63.0000 | 0.1812 |
| nn_moderate_64_xg | 0.7414 | 0.9191 | 0.9099 | 56.0000 | 0.1777 |
| nn_two_layer_32_16_xg | 0.7432 | 0.9258 | 0.9318 | 54.0000 | 0.1826 |
| nn_tiny_16_xg | 0.7561 | 0.9281 | 0.9260 | 75.0000 | 0.1719 |
| nn_tiny_8_xg | 0.7862 | 0.9294 | 0.9288 | 109.0000 | 0.1432 |
| nn_two_layer_64_32_xg | 0.7255 | 0.9451 | 0.9511 | 54.0000 | 0.2195 |
| nn_logistic_xg | 0.7758 | 0.9452 | 0.9411 | 118.0000 | 0.1694 |

Overconfidence diagnostics:

| experiment_name | avg_max_confidence | high_confidence_wrong_count | actual_probability_below_05_count | actual_probability_below_10_count |
| --- | --- | --- | --- | --- |
| nn_logistic_xg | 0.5977 | 11 | 6 | 19 |
| nn_moderate_32_xg | 0.5775 | 9 | 4 | 12 |
| nn_moderate_64_xg | 0.5683 | 7 | 3 | 13 |
| nn_tiny_16_xg | 0.5842 | 10 | 2 | 16 |
| nn_tiny_8_xg | 0.5926 | 2 | 2 | 10 |
| nn_two_layer_32_16_xg | 0.5775 | 11 | 5 | 20 |
| nn_two_layer_64_32_xg | 0.5815 | 11 | 2 | 16 |

The best NN by log loss, `nn_tiny_8_xg`, averaged max confidence 0.5926 across seeds, with 2 high-confidence wrong predictions in aggregate and 10 instances where the actual outcome probability was below 10%. Wider/deeper models generally did not reduce these risk signals enough to compensate for worse log loss.

## Recommendation

Do not promote a neural network challenger from this corrected xG Phase 3 run. `champion_dc_xg` remains the best model by log loss and Brier score, and no NN beats `improved_logistic_regression` on the priority metrics. The architecture ladder suggests that extra width/depth adds variance and mild overfit pressure rather than robust probability-quality gains on the current 109-fixture evaluation window. Keep the NN work as research-only unless future data volume or feature changes produce a clear log-loss and Brier improvement, not merely an accuracy bump.
