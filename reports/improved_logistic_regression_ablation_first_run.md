# Improved Logistic Regression Ablation Report

Feature-group ablations for Phase 8B-1 over the same 2025-10-01 to 2026-05-16 window. Lower Brier/log loss are better; higher accuracy is better.

## Ablation Summary

The xG feature group is the best ablation and is used as the registered
`improved_logistic_regression` default. The full feature set underperforms,
which is consistent with the small one-season WSL sample and supports keeping
the Phase 8B-1 feature set conservative.

## Model Comparison

| rank | model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | improved_logistic_xg | 109 | 0.5384 | 0.9356 | 0.5780 |
| 2 | improved_logistic_base | 109 | 0.5529 | 0.9496 | 0.5688 |
| 3 | improved_logistic_opponent | 109 | 0.5530 | 0.9521 | 0.5688 |
| 4 | improved_logistic_full | 109 | 0.5786 | 1.0112 | 0.5505 |
| 5 | improved_logistic_form | 109 | 0.5878 | 1.0145 | 0.5321 |

## Calibration

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 28 | 0.3780 | 0.3571 | 0.0209 | False |
| 40%-60% | 285 | 0.5018 | 0.4421 | 0.0597 | False |
| 60%-80% | 206 | 0.6771 | 0.6990 | -0.0219 | False |
| 80%-100% | 26 | 0.8324 | 0.9615 | -0.1291 | False |

## Confidence Buckets

| bucket | count | mean_confidence | accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| low | 28 | 0.3780 | 0.3571 | 0.0209 | False |
| medium | 285 | 0.5018 | 0.4421 | 0.0597 | False |
| high | 232 | 0.6945 | 0.7284 | -0.0339 | False |

## Worst Misses

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| improved_logistic_full | Leicester City | Everton | D | A | 0.6856 | 0.0218 | 3.8249 |
| improved_logistic_full | Chelsea | Everton | A | H | 0.7525 | 0.0298 | 3.5128 |
| improved_logistic_form | Leicester City | Everton | D | H | 0.5231 | 0.0444 | 3.1147 |
| improved_logistic_opponent | Leicester City | Everton | D | H | 0.6464 | 0.0527 | 2.9440 |
| improved_logistic_xg | Leicester City | Everton | D | A | 0.5309 | 0.0538 | 2.9224 |

## Best High-Confidence Correct

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| improved_logistic_full | Manchester City | Leicester City | H | H | 0.9131 | 0.9131 | 0.0909 |
| improved_logistic_xg | Arsenal | Leicester City | H | H | 0.8754 | 0.8754 | 0.1331 |
| improved_logistic_xg | Manchester City | Leicester City | H | H | 0.8555 | 0.8555 | 0.1561 |
| improved_logistic_opponent | London City Lionesses | Liverpool | H | H | 0.8555 | 0.8555 | 0.1561 |
| improved_logistic_full | Arsenal | Leicester City | H | H | 0.8498 | 0.8498 | 0.1628 |

## Favourite Breakdown

| predicted_outcome | confidence_bucket | n | accuracy | mean_confidence | mean_actual_probability | mean_log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| A | high | 79 | 0.6203 | 0.6822 | 0.4801 | 0.9944 |
| A | low | 11 | 0.2727 | 0.3844 | 0.3349 | 1.1095 |
| A | medium | 127 | 0.4488 | 0.4923 | 0.3665 | 1.0923 |
| D | low | 7 | 0.0000 | 0.3730 | 0.3119 | 1.1702 |
| D | medium | 5 | 0.0000 | 0.4547 | 0.2964 | 1.2819 |
| H | high | 153 | 0.7843 | 0.7008 | 0.5969 | 0.6865 |
| H | low | 10 | 0.7000 | 0.3745 | 0.3655 | 1.0082 |
| H | medium | 153 | 0.4510 | 0.5112 | 0.3646 | 1.1168 |
