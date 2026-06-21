# Champion Diagnostics Report

## Executive Summary

`champion_dc_xg` remains the reference model on 109 evaluated matches: Brier 0.5052, log loss 0.8669, accuracy 0.6239.
Nearest challenger: logistic_regression (Brier 0.5562, log loss 0.9560, accuracy 0.5780).
This report diagnoses where the champion wins and fails; it does not change model behaviour.

## Champion Headline Metrics

| model_name | n_matches | brier_score | log_loss | accuracy | rank |
| --- | --- | --- | --- | --- | --- |
| champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 | 1 |

## Why Champion Remains The Reference Model

| rank | model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 |
| 2 | logistic_regression | 109 | 0.5562 | 0.9560 | 0.5780 |
| 3 | elo_baseline | 109 | 0.5676 | 0.9596 | 0.5596 |
| 4 | naive_outcome_rate | 109 | 0.6401 | 1.0576 | 0.4587 |

## High-Confidence Misses

| match_date | round | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-12-07 | R10 | Chelsea | Everton | A | H | 0.8531 | 0.0450 | 3.1021 |
| 2025-11-16 | R9 | Liverpool | Chelsea | D | A | 0.7920 | 0.1470 | 1.9173 |
| 2026-03-15 | R17 | Aston Villa | Manchester City | D | A | 0.7550 | 0.1490 | 1.9038 |
| 2025-11-02 | R7 | Aston Villa | Everton | D | H | 0.7550 | 0.1610 | 1.8264 |
| 2025-11-08 | R8 | Manchester United | Aston Villa | A | H | 0.6850 | 0.1530 | 1.8773 |
| 2025-11-16 | R9 | Tottenham Hotspur | Arsenal | D | A | 0.6637 | 0.2172 | 1.5269 |
| 2026-05-06 | R16 | Brighton | Arsenal | D | A | 0.6386 | 0.2643 | 1.3308 |

## High-Confidence Correct Predictions

| match_date | round | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-29 | R14 | Arsenal | Leicester City | H | H | 0.9189 | 0.9189 | 0.0846 |
| 2026-01-11 | R12 | Manchester City | Everton | H | H | 0.9080 | 0.9080 | 0.0965 |
| 2026-02-13 | R16 | Manchester City | Leicester City | H | H | 0.8940 | 0.8940 | 0.1120 |
| 2026-05-13 | R21 | Arsenal | Everton | H | H | 0.8651 | 0.8651 | 0.1449 |
| 2025-11-01 | R7 | Manchester City | West Ham United | H | H | 0.8620 | 0.8620 | 0.1485 |
| 2025-12-14 | R11 | Manchester City | Aston Villa | H | H | 0.8620 | 0.8620 | 0.1485 |
| 2026-03-21 | R18 | Arsenal | West Ham United | H | H | 0.8600 | 0.8600 | 0.1508 |
| 2026-03-28 | R19 | Arsenal | Tottenham Hotspur | H | H | 0.8500 | 0.8500 | 0.1625 |
| 2026-03-21 | R18 | Manchester City | Tottenham Hotspur | H | H | 0.8302 | 0.8302 | 0.1861 |
| 2025-10-12 | R6 | Liverpool | Manchester City | A | A | 0.8300 | 0.8300 | 0.1863 |

## Confidence-Band Performance

| confidence_bucket | n | accuracy | mean_confidence | mean_actual_probability | mean_log_loss |
| --- | --- | --- | --- | --- | --- |
| high | 43 | 0.8372 | 0.7544 | 0.6613 | 0.5502 |
| low | 15 | 0.4667 | 0.3697 | 0.3397 | 1.0855 |
| medium | 51 | 0.4902 | 0.4915 | 0.3695 | 1.0696 |

## Draw And Favourite Behaviour

| favourite_type | n | accuracy | mean_confidence | mean_actual_probability | mean_log_loss |
| --- | --- | --- | --- | --- | --- |
| away_favourite | 42 | 0.5952 | 0.5723 | 0.4624 | 0.8924 |
| home_favourite | 63 | 0.6508 | 0.5965 | 0.5014 | 0.8361 |
| predicted_draw | 4 | 0.5000 | 0.3598 | 0.3403 | 1.0831 |

## Team-Level Error Patterns

| team | n | error_rate | mean_log_loss | mean_confidence |
| --- | --- | --- | --- | --- |
| Tottenham Hotspur | 18 | 0.5556 | 0.9921 | 0.5421 |
| Brighton | 18 | 0.5000 | 1.0247 | 0.5038 |
| West Ham United | 18 | 0.5000 | 0.9439 | 0.5386 |
| Aston Villa | 19 | 0.4737 | 1.0439 | 0.5464 |
| Liverpool | 19 | 0.4737 | 0.9105 | 0.5419 |
| London City Lionesses | 18 | 0.3889 | 0.9309 | 0.5144 |
| Everton | 18 | 0.3333 | 0.9524 | 0.6085 |
| Manchester United | 18 | 0.3333 | 0.8918 | 0.5566 |
| Chelsea | 18 | 0.2778 | 0.8422 | 0.6413 |
| Arsenal | 18 | 0.2778 | 0.6273 | 0.6699 |

## Round/Fold Instability

| round_or_fold | n | accuracy | mean_confidence | mean_log_loss |
| --- | --- | --- | --- | --- |
| R5 | 6 | 0.5000 | 0.4333 | 1.2026 |
| R10 | 6 | 0.5000 | 0.6305 | 1.1046 |
| R9 | 6 | 0.6667 | 0.5517 | 1.1006 |
| R17 | 6 | 0.5000 | 0.5958 | 1.0838 |
| R15 | 6 | 0.5000 | 0.5361 | 1.0266 |
| R8 | 6 | 0.3333 | 0.5650 | 1.0212 |
| R20 | 6 | 0.5000 | 0.5551 | 1.0044 |
| R11 | 6 | 0.5000 | 0.5982 | 0.9145 |
| R13 | 6 | 0.6667 | 0.4687 | 0.8420 |
| R3 | 1 | 1.0000 | 0.4380 | 0.8255 |
| R14 | 6 | 0.5000 | 0.5609 | 0.7924 |
| R19 | 6 | 0.6667 | 0.5564 | 0.7860 |
| R7 | 6 | 0.8333 | 0.6287 | 0.7590 |
| R21 | 6 | 0.8333 | 0.6180 | 0.7481 |
| R12 | 6 | 0.6667 | 0.6161 | 0.7405 |
| R16 | 6 | 0.6667 | 0.6447 | 0.7082 |
| R18 | 6 | 0.6667 | 0.5847 | 0.7022 |
| R22 | 6 | 0.8333 | 0.6322 | 0.5653 |
| R6 | 6 | 0.8333 | 0.6594 | 0.5093 |

## Champion Versus Nearest Challenger

| challenger_model | paired_matches | champion_only_correct | challenger_only_correct | both_correct | both_wrong | mean_log_loss_delta_champion_minus_challenger | mean_actual_probability_delta_champion_minus_challenger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 109 | 10 | 5 | 58 | 36 | -0.0891 | 0.0326 |

## Limitations

- Diagnostics use one generated comparison artefact only.
- The champion sample contains 109 evaluated matches.
- Rows are evaluated from completed fixtures already present in the comparison artefact.

## Specific Next Modelling Recommendations

- Keep champion_dc_xg as the reference model until a challenger beats it on the same folds.
- Test calibration improvements before adding new model families, because confidence gaps are visible in bands.
- Review team-specific residuals for Tottenham Hotspur, Brighton, West Ham United before changing the model form.
