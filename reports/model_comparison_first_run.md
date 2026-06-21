# WSL Model Evaluation Report

## Model Comparison

| rank | model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 |
| 2 | logistic_regression | 109 | 0.5562 | 0.9560 | 0.5780 |
| 3 | elo_baseline | 109 | 0.5676 | 0.9596 | 0.5596 |
| 4 | naive_outcome_rate | 109 | 0.6401 | 1.0576 | 0.4587 |

## Calibration

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 31 | 0.3781 | 0.4839 | -0.1058 | False |
| 40%-60% | 307 | 0.4737 | 0.4886 | -0.0149 | False |
| 60%-80% | 82 | 0.6887 | 0.7561 | -0.0674 | False |
| 80%-100% | 16 | 0.8536 | 0.9375 | -0.0839 | False |

## Confidence Buckets

| bucket | count | mean_confidence | accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| low | 31 | 0.3781 | 0.4839 | -0.1058 | False |
| medium | 307 | 0.4737 | 0.4886 | -0.0149 | False |
| high | 98 | 0.7156 | 0.7857 | -0.0701 | False |

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
| A | high | 33 | 0.6970 | 0.7061 | 0.5507 | 0.7752 |
| A | low | 10 | 0.6000 | 0.3767 | 0.3490 | 1.0592 |
| A | medium | 62 | 0.5000 | 0.4751 | 0.3662 | 1.0697 |
| D | low | 4 | 0.5000 | 0.3598 | 0.3403 | 1.0831 |
| H | high | 65 | 0.8308 | 0.7205 | 0.6300 | 0.6131 |
| H | low | 17 | 0.4118 | 0.3832 | 0.3445 | 1.0755 |
| H | medium | 245 | 0.4857 | 0.4734 | 0.3774 | 1.0351 |
