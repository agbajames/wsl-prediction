# WSL Model Evaluation Report

## Poisson Challenger First-Run Summary

The `poisson_regression` challenger evaluated 109 matches from the local WSL
CSV. It did not beat the production champion or the stronger
`regularised_team_strength` challenger on this first run.

Headline result:

- `champion_dc_xg` remains best overall: Brier 0.5052, log loss 0.8669,
  accuracy 0.6239.
- `regularised_team_strength` remains the strongest challenger: Brier 0.5256,
  log loss 0.9001, accuracy 0.5963.
- `poisson_regression` ranks third by log loss: Brier 0.5585, log loss 0.9426,
  accuracy 0.5413.
- `poisson_regression` beats `elo_baseline` and `naive_outcome_rate` on Brier
  and log loss, and beats `logistic_regression` on log loss only.

Interpretation: the regularised Poisson scoring-rate model is a useful
interpretable challenger, but the first local comparison suggests that simple
team attack/defence regression is not enough to close the gap to the xG-based
champion or the smoothed team-strength challenger. Results should still be
treated cautiously because this window covers one WSL season and 109 evaluated
matches.

## Model Comparison

| rank | model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 |
| 2 | regularised_team_strength | 109 | 0.5256 | 0.9001 | 0.5963 |
| 3 | poisson_regression | 109 | 0.5585 | 0.9426 | 0.5413 |
| 4 | logistic_regression | 109 | 0.5562 | 0.9560 | 0.5780 |
| 5 | elo_baseline | 109 | 0.5676 | 0.9596 | 0.5596 |
| 6 | naive_outcome_rate | 109 | 0.6401 | 1.0576 | 0.4587 |

## Calibration

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 66 | 0.3798 | 0.4697 | -0.0899 | False |
| 40%-60% | 429 | 0.4801 | 0.4802 | -0.0001 | False |
| 60%-80% | 132 | 0.6858 | 0.7803 | -0.0945 | False |
| 80%-100% | 27 | 0.8467 | 0.9630 | -0.1163 | False |

## Confidence Buckets

| bucket | count | mean_confidence | accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| low | 66 | 0.3798 | 0.4697 | -0.0899 | False |
| medium | 429 | 0.4801 | 0.4802 | -0.0001 | False |
| high | 159 | 0.7131 | 0.8113 | -0.0982 | False |

## Worst Misses

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| champion_dc_xg | Chelsea | Everton | A | H | 0.8531 | 0.0450 | 3.1021 |
| logistic_regression | Leicester City | Everton | D | H | 0.6422 | 0.0612 | 2.7929 |
| logistic_regression | West Ham United | Leicester City | D | H | 0.5243 | 0.0786 | 2.5437 |
| poisson_regression | West Ham United | Everton | H | A | 0.7707 | 0.0958 | 2.3450 |
| logistic_regression | Manchester United | Aston Villa | A | H | 0.6184 | 0.0991 | 2.3117 |

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
| A | high | 60 | 0.7333 | 0.7008 | 0.5579 | 0.7536 |
| A | low | 25 | 0.4800 | 0.3802 | 0.3487 | 1.0611 |
| A | medium | 120 | 0.4417 | 0.4803 | 0.3571 | 1.0910 |
| D | low | 4 | 0.5000 | 0.3598 | 0.3403 | 1.0831 |
| H | high | 99 | 0.8586 | 0.7205 | 0.6463 | 0.5602 |
| H | low | 37 | 0.4595 | 0.3817 | 0.3444 | 1.0756 |
| H | medium | 309 | 0.4951 | 0.4800 | 0.3813 | 1.0264 |
