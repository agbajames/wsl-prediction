# WSL Model Evaluation Report

## Improved Logistic First-Run Summary

`improved_logistic_regression` uses the conservative xG feature group selected
from the Phase 8B-1 ablation. It improves the original logistic model on Brier
score and log loss, while matching its accuracy.

- `improved_logistic_regression`: Brier 0.5384, log loss 0.9356, accuracy 0.5780.
- `logistic_regression`: Brier 0.5562, log loss 0.9560, accuracy 0.5780.
- `regularised_team_strength` remains the strongest standalone non-champion challenger.
- `champion_dc_xg` remains the operational/reference model.

Interpretation: richer leakage-safe xG features help the logistic challenger,
but the model still does not close the gap to the champion-family candidates.
It should remain offline and feed into later Phase 8B feature work.

## Model Comparison

| rank | model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 |
| 2 | regularised_team_strength | 109 | 0.5256 | 0.9001 | 0.5963 |
| 3 | improved_logistic_regression | 109 | 0.5384 | 0.9356 | 0.5780 |
| 4 | poisson_regression | 109 | 0.5585 | 0.9426 | 0.5413 |
| 5 | logistic_regression | 109 | 0.5562 | 0.9560 | 0.5780 |
| 6 | elo_baseline | 109 | 0.5676 | 0.9596 | 0.5596 |
| 7 | naive_outcome_rate | 109 | 0.6401 | 1.0576 | 0.4587 |

## Calibration

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 67 | 0.3801 | 0.4627 | -0.0826 | False |
| 40%-60% | 491 | 0.4838 | 0.4705 | 0.0133 | False |
| 60%-80% | 170 | 0.6870 | 0.7824 | -0.0954 | False |
| 80%-100% | 35 | 0.8445 | 0.9714 | -0.1269 | False |

## Confidence Buckets

| bucket | count | mean_confidence | accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| low | 67 | 0.3801 | 0.4627 | -0.0826 | False |
| medium | 491 | 0.4838 | 0.4705 | 0.0133 | False |
| high | 205 | 0.7139 | 0.8146 | -0.1007 | False |

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
| A | high | 74 | 0.7297 | 0.7032 | 0.5569 | 0.7619 |
| A | low | 26 | 0.4615 | 0.3809 | 0.3495 | 1.0586 |
| A | medium | 151 | 0.4437 | 0.4848 | 0.3566 | 1.1032 |
| D | low | 4 | 0.5000 | 0.3598 | 0.3403 | 1.0831 |
| D | medium | 1 | 0.0000 | 0.4180 | 0.4128 | 0.8849 |
| H | high | 131 | 0.8626 | 0.7199 | 0.6468 | 0.5627 |
| H | low | 37 | 0.4595 | 0.3817 | 0.3444 | 1.0756 |
| H | medium | 339 | 0.4838 | 0.4835 | 0.3778 | 1.0389 |
