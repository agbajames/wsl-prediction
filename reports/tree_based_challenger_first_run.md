# WSL Model Evaluation Report

## Tree-Based Challenger First-Run Summary

`random_forest` is a conservative offline tree-based challenger using the same
leakage-safe xG feature group as `improved_logistic_regression`.

- `random_forest`: Brier 0.5491, log loss 0.9385, accuracy 0.5872.
- It beats the original `logistic_regression` on log loss and accuracy, but not Brier score.
- It beats `poisson_regression` on Brier score, log loss, and accuracy.
- It does not beat `improved_logistic_regression`, `regularised_team_strength`, or `champion_dc_xg` on probability-quality metrics.

Interpretation: shallow tree interactions help accuracy relative to the logistic
family baseline, but the first run does not justify treating the tree model as a
leading challenger. It remains useful Phase 8B evidence and should stay
offline/evaluation-only.

## Model Comparison

| rank | model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 |
| 2 | regularised_team_strength | 109 | 0.5256 | 0.9001 | 0.5963 |
| 3 | improved_logistic_regression | 109 | 0.5384 | 0.9356 | 0.5780 |
| 4 | random_forest | 109 | 0.5491 | 0.9385 | 0.5872 |
| 5 | poisson_regression | 109 | 0.5585 | 0.9426 | 0.5413 |
| 6 | logistic_regression | 109 | 0.5562 | 0.9560 | 0.5780 |
| 7 | elo_baseline | 109 | 0.5676 | 0.9596 | 0.5596 |
| 8 | naive_outcome_rate | 109 | 0.6401 | 1.0576 | 0.4587 |

## Calibration

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 78 | 0.3805 | 0.4615 | -0.0810 | False |
| 40%-60% | 550 | 0.4854 | 0.4709 | 0.0145 | False |
| 60%-80% | 209 | 0.6820 | 0.7847 | -0.1027 | False |
| 80%-100% | 35 | 0.8445 | 0.9714 | -0.1269 | False |

## Confidence Buckets

| bucket | count | mean_confidence | accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| low | 78 | 0.3805 | 0.4615 | -0.0810 | False |
| medium | 550 | 0.4854 | 0.4709 | 0.0145 | False |
| high | 244 | 0.7053 | 0.8115 | -0.1062 | False |

## Worst Misses

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| champion_dc_xg | Chelsea | Everton | A | H | 0.8531 | 0.0450 | 3.1021 |
| improved_logistic_regression | Leicester City | Everton | D | A | 0.5309 | 0.0538 | 2.9224 |
| logistic_regression | Leicester City | Everton | D | H | 0.6422 | 0.0612 | 2.7929 |
| improved_logistic_regression | Aston Villa | Leicester City | D | H | 0.6977 | 0.0622 | 2.7773 |
| improved_logistic_regression | Chelsea | Everton | A | H | 0.7952 | 0.0626 | 2.7713 |

## Best High-Confidence Correct

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| champion_dc_xg | Arsenal | Leicester City | H | H | 0.9189 | 0.9189 | 0.0846 |
| champion_dc_xg | Manchester City | Everton | H | H | 0.9080 | 0.9080 | 0.0965 |
| poisson_regression | Manchester City | Liverpool | H | H | 0.8983 | 0.8983 | 0.1073 |
| poisson_regression | Manchester City | Leicester City | H | H | 0.8961 | 0.8961 | 0.1097 |
| champion_dc_xg | Manchester City | Leicester City | H | H | 0.8940 | 0.8940 | 0.1120 |

## Favourite Breakdown

| predicted_outcome | confidence_bucket | n | accuracy | mean_confidence | mean_actual_probability | mean_log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| A | high | 91 | 0.7253 | 0.6956 | 0.5487 | 0.7750 |
| A | low | 32 | 0.4688 | 0.3820 | 0.3493 | 1.0597 |
| A | medium | 167 | 0.4371 | 0.4834 | 0.3549 | 1.1089 |
| D | low | 5 | 0.4000 | 0.3664 | 0.3244 | 1.1352 |
| D | medium | 1 | 0.0000 | 0.4180 | 0.4128 | 0.8849 |
| H | high | 153 | 0.8627 | 0.7110 | 0.6403 | 0.5648 |
| H | low | 41 | 0.4634 | 0.3812 | 0.3436 | 1.0789 |
| H | medium | 382 | 0.4869 | 0.4865 | 0.3783 | 1.0389 |
