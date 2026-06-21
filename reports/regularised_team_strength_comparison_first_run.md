# WSL Model Evaluation Report

## Model Comparison

| rank | model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 |
| 2 | regularised_team_strength | 109 | 0.5256 | 0.9001 | 0.5963 |
| 3 | logistic_regression | 109 | 0.5562 | 0.9560 | 0.5780 |
| 4 | elo_baseline | 109 | 0.5676 | 0.9596 | 0.5596 |
| 5 | naive_outcome_rate | 109 | 0.6401 | 1.0576 | 0.4587 |

## Calibration

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 52 | 0.3796 | 0.4808 | -0.1012 | False |
| 40%-60% | 370 | 0.4773 | 0.4919 | -0.0146 | False |
| 60%-80% | 103 | 0.6859 | 0.7864 | -0.1005 | False |
| 80%-100% | 20 | 0.8454 | 0.9500 | -0.1046 | False |

## Confidence Buckets

| bucket | count | mean_confidence | accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| low | 52 | 0.3796 | 0.4808 | -0.1012 | False |
| medium | 370 | 0.4773 | 0.4919 | -0.0146 | False |
| high | 123 | 0.7118 | 0.8130 | -0.1012 | False |

## Worst Misses

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| champion_dc_xg | Chelsea | Everton | A | H | 0.8531 | 0.0450 | 3.1021 |
| logistic_regression | Leicester City | Everton | D | H | 0.6422 | 0.0612 | 2.7929 |
| logistic_regression | West Ham United | Leicester City | D | H | 0.5243 | 0.0786 | 2.5437 |
| logistic_regression | Manchester United | Aston Villa | A | H | 0.6184 | 0.0991 | 2.3117 |
| logistic_regression | Brighton | Manchester City | H | A | 0.6629 | 0.1054 | 2.2497 |

## Best High-Confidence Correct

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| champion_dc_xg | Arsenal | Leicester City | H | H | 0.9189 | 0.9189 | 0.0846 |
| champion_dc_xg | Manchester City | Everton | H | H | 0.9080 | 0.9080 | 0.0965 |
| champion_dc_xg | Manchester City | Leicester City | H | H | 0.8940 | 0.8940 | 0.1120 |
| champion_dc_xg | Arsenal | Everton | H | H | 0.8651 | 0.8651 | 0.1449 |
| logistic_regression | London City Lionesses | Liverpool | H | H | 0.8646 | 0.8646 | 0.1455 |

## Favourite Breakdown

| predicted_outcome | confidence_bucket | n | accuracy | mean_confidence | mean_actual_probability | mean_log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| A | high | 41 | 0.7317 | 0.6932 | 0.5562 | 0.7434 |
| A | low | 18 | 0.5000 | 0.3793 | 0.3459 | 1.0702 |
| A | medium | 88 | 0.5000 | 0.4793 | 0.3718 | 1.0520 |
| D | low | 4 | 0.5000 | 0.3598 | 0.3403 | 1.0831 |
| H | high | 82 | 0.8537 | 0.7211 | 0.6424 | 0.5749 |
| H | low | 30 | 0.4667 | 0.3825 | 0.3472 | 1.0677 |
| H | medium | 282 | 0.4894 | 0.4767 | 0.3790 | 1.0314 |

## Interpretation

`regularised_team_strength` is the strongest challenger tested so far. It ranks second overall, ahead of `logistic_regression`, `elo_baseline`, and `naive_outcome_rate` on Brier score, log loss, and accuracy.

It does not beat `champion_dc_xg`. The champion remains the reference model after this run: Brier 0.5052 vs 0.5256, log loss 0.8669 vs 0.9001, and accuracy 0.6239 vs 0.5963.

## Limitations

- This is one WSL season slice with 109 evaluated matches per model.
- The new challenger uses an independent Poisson grid and does not include Dixon-Coles low-score correlation or penalty-specific modelling.
- The result is useful evidence for Phase 8A, but not enough for a production model decision.
