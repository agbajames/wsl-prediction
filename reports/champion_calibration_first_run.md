# Champion Calibration Experiment

## Executive Summary

This offline experiment fits a calibration layer for `champion_dc_xg` on 66 earlier matches and evaluates it on 43 later matches.
Calibrated minus original deltas on the trial split: Brier +0.0028, log loss +0.0120, accuracy -0.0233.
Recommendation: do not replace the champion reference from this experiment; keep testing calibration variants offline.

## Methodology

- Fit temperature scaling plus base-rate shrinkage on earlier folds; evaluate on later folds.
- Calibration folds: fold_001, fold_002, fold_003, fold_004, fold_005, fold_006, fold_007, fold_008, fold_009, fold_010, fold_011
- Trial folds: fold_012, fold_013, fold_014, fold_015, fold_016, fold_017, fold_018, fold_019
- The underlying champion model, API, dashboard, and production outputs are unchanged.

## Original Champion Vs Calibrated Champion

| model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- |
| original_champion | 43 | 0.4503 | 0.7831 | 0.6744 |
| calibrated_champion | 43 | 0.4531 | 0.7951 | 0.6512 |

## Calibration Parameters

| method | temperature | shrinkage | base_home | base_draw | base_away |
| --- | --- | --- | --- | --- | --- |
| temperature_scaling_with_base_rate_shrinkage | 0.7250 | 0.2100 | 0.4545 | 0.1970 | 0.3485 |

## Overall Metrics

| brier_score | log_loss | accuracy |
| --- | --- | --- |
| 0.0028 | 0.0120 | -0.0233 |

## Calibration Bands

### Original Champion

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 4 | 0.3783 | 0.7500 | -0.3717 | False |
| 40%-60% | 19 | 0.4916 | 0.4211 | 0.0705 | False |
| 60%-80% | 14 | 0.7121 | 0.8571 | -0.1450 | False |
| 80%-100% | 6 | 0.8697 | 1.0000 | -0.1303 | False |

### Calibrated Champion

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 2 | 0.3741 | 0.5000 | -0.1259 | False |
| 40%-60% | 18 | 0.4926 | 0.3889 | 0.1037 | False |
| 60%-80% | 17 | 0.7072 | 0.8235 | -0.1163 | False |
| 80%-100% | 6 | 0.8395 | 1.0000 | -0.1605 | False |

## High-Confidence Behaviour

| model_name | threshold | count | share | accuracy | mean_confidence |
| --- | --- | --- | --- | --- | --- |
| original_champion | 0.6000 | 20 | 0.4651 | 0.9000 | 0.7594 |
| calibrated_champion | 0.6000 | 23 | 0.5349 | 0.8696 | 0.7417 |

## Worst Misses After Calibration

| match_date | round | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-15 | R17 | Aston Villa | Manchester City | D | A | 0.7514 | 0.1137 | 2.1743 |
| 2026-05-04 | R21 | Aston Villa | West Ham United | A | H | 0.5944 | 0.1608 | 1.8274 |
| 2026-04-26 | R20 | Liverpool | West Ham United | A | H | 0.5309 | 0.1638 | 1.8088 |
| 2026-03-15 | R17 | Tottenham Hotspur | Everton | A | H | 0.6237 | 0.1767 | 1.7330 |
| 2026-05-06 | R16 | Brighton | Arsenal | D | A | 0.6496 | 0.2120 | 1.5510 |
| 2026-04-25 | R20 | Brighton | Manchester City | H | A | 0.5303 | 0.2348 | 1.4489 |
| 2026-04-26 | R20 | Tottenham Hotspur | Manchester United | D | H | 0.4093 | 0.2362 | 1.4429 |
| 2026-03-28 | R19 | Everton | Liverpool | A | H | 0.4459 | 0.2413 | 1.4219 |
| 2026-03-18 | R17 | West Ham United | Manchester United | D | A | 0.5028 | 0.2527 | 1.3754 |
| 2026-03-21 | R18 | London City Lionesses | Chelsea | D | A | 0.4685 | 0.2561 | 1.3623 |

## Limitations

- This is one offline experiment over the first comparison artefact.
- Calibration parameters are fit on a small WSL sample, so they should be treated as hypotheses.
- The trial split is later folds only, which is safer than evaluating on the same rows used for calibration.

## Recommendation

Recommendation: do not replace the champion reference from this experiment; keep testing calibration variants offline.

- Calibration did not improve Brier score or log loss on this trial split; keep the original champion as reference.
- Accuracy also fell on this trial split, so this calibrated variant is not a production-testing candidate yet.
