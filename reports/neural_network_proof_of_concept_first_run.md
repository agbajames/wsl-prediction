# WSL Model Evaluation Report

## Neural Network Proof-Of-Concept First-Run Summary

`neural_network` is a research-only tiny MLP over the same leakage-safe xG
feature group used by the Phase 8B feature models.

- `neural_network`: Brier 0.5567, log loss 0.9762, accuracy 0.6055.
- It has higher accuracy than `improved_logistic_regression`, `random_forest`,
  and `regularised_team_strength` in this run.
- It does not beat `improved_logistic_regression`, `random_forest`,
  `regularised_team_strength`, or `champion_dc_xg` on probability-quality
  metrics.
- The worst misses are highly confident, which is a small-sample overfitting
  warning.

Interpretation: the neural net learned some useful class-boundary signal, but
its probability calibration is not good enough to treat it as a serious model
candidate. This result supports keeping Phase 8C research-only and prioritising
Phase 8D blending plus Phase 8E shadow testing before any production decision.

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
| 8 | neural_network | 109 | 0.5567 | 0.9762 | 0.6055 |
| 9 | naive_outcome_rate | 109 | 0.6401 | 1.0576 | 0.4587 |

## Calibration

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 80 | 0.3802 | 0.4625 | -0.0823 | False |
| 40%-60% | 593 | 0.4877 | 0.4739 | 0.0138 | False |
| 60%-80% | 251 | 0.6842 | 0.7570 | -0.0728 | False |
| 80%-100% | 57 | 0.8569 | 0.8947 | -0.0378 | False |

## Confidence Buckets

| bucket | count | mean_confidence | accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| low | 80 | 0.3802 | 0.4625 | -0.0823 | False |
| medium | 593 | 0.4877 | 0.4739 | 0.0138 | False |
| high | 308 | 0.7162 | 0.7825 | -0.0663 | False |

## Worst Misses

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neural_network | Chelsea | Everton | A | H | 0.8987 | 0.0218 | 3.8269 |
| neural_network | Tottenham Hotspur | Brighton | H | A | 0.8960 | 0.0334 | 3.3992 |
| neural_network | Aston Villa | Leicester City | D | H | 0.8914 | 0.0338 | 3.3880 |
| champion_dc_xg | Chelsea | Everton | A | H | 0.8531 | 0.0450 | 3.1021 |
| neural_network | Leicester City | Everton | D | A | 0.8233 | 0.0460 | 3.0790 |

## Best High-Confidence Correct

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neural_network | London City Lionesses | Liverpool | H | H | 0.9595 | 0.9595 | 0.0413 |
| neural_network | Manchester City | West Ham United | H | H | 0.9350 | 0.9350 | 0.0672 |
| neural_network | Arsenal | Liverpool | H | H | 0.9312 | 0.9312 | 0.0713 |
| champion_dc_xg | Arsenal | Leicester City | H | H | 0.9189 | 0.9189 | 0.0846 |
| champion_dc_xg | Manchester City | Everton | H | H | 0.9080 | 0.9080 | 0.0965 |

## Favourite Breakdown

| predicted_outcome | confidence_bucket | n | accuracy | mean_confidence | mean_actual_probability | mean_log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| A | high | 117 | 0.6838 | 0.7037 | 0.5273 | 0.8735 |
| A | low | 34 | 0.4706 | 0.3810 | 0.3493 | 1.0592 |
| A | medium | 189 | 0.4497 | 0.4851 | 0.3592 | 1.0948 |
| D | low | 5 | 0.4000 | 0.3664 | 0.3244 | 1.1352 |
| D | medium | 1 | 0.0000 | 0.4180 | 0.4128 | 0.8849 |
| H | high | 191 | 0.8429 | 0.7238 | 0.6384 | 0.5987 |
| H | low | 41 | 0.4634 | 0.3812 | 0.3436 | 1.0789 |
| H | medium | 403 | 0.4864 | 0.4891 | 0.3780 | 1.0418 |
