# Time-Decay and xG-Weighting Variant Experiment

## First-Run Summary

This offline experiment keeps `champion_dc_xg` as the unchanged reference, includes
`dc_fit_rho_each_fold` as the best Phase 8A-3 comparison reference, and tests a small
predeclared grid of time-decay and xG-weighting variants.

Best-ranked variant: `dc_fit_rho_each_fold` versus champion reference `champion_dc_xg`.

- Champion reference: Brier 0.5052, log loss 0.8669, accuracy 0.6239.
- Best ranked row: `dc_fit_rho_each_fold` with Brier 0.5027, log loss 0.8623, accuracy 0.6147.

## Predeclared Variants

| model_name | change |
| --- | --- |
| champion_dc_xg | Frozen champion reference configuration: 60-day decay, fixed rho -0.13. |
| dc_fit_rho_each_fold | Best Phase 8A-3 comparison reference: fit rho inside each training fold. |
| txg_decay_45d | Slightly shorter recency half-life: 45 days instead of 60. |
| txg_decay_90d | Slightly longer recency half-life: 90 days instead of 60. |
| txg_alpha_025 | Moderately stronger xG team-strength ridge shrinkage: alpha 0.25. |
| txg_xg_pseudocount_010 | Conservative np_xG floor/shrinkage: xG pseudocount 0.10. |
| txg_conservative_weighting | Combined conservative xG weighting: alpha 0.25, xG pseudocount 0.10, penalty shrinkage 10. |

## Model Comparison

| rank | model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | dc_fit_rho_each_fold | 109 | 0.5027 | 0.8623 | 0.6147 |
| 2 | txg_xg_pseudocount_010 | 109 | 0.5038 | 0.8649 | 0.6330 |
| 3 | champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 |
| 4 | txg_decay_90d | 109 | 0.5055 | 0.8672 | 0.6147 |
| 5 | txg_alpha_025 | 109 | 0.5056 | 0.8677 | 0.6239 |
| 6 | txg_conservative_weighting | 109 | 0.5058 | 0.8682 | 0.6239 |
| 7 | txg_decay_45d | 109 | 0.5061 | 0.8684 | 0.6239 |

## Calibration

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 101 | 0.3700 | 0.4752 | -0.1052 | False |
| 40%-60% | 360 | 0.4904 | 0.4833 | 0.0071 | False |
| 60%-80% | 213 | 0.7068 | 0.8075 | -0.1007 | False |
| 80%-100% | 89 | 0.8551 | 0.9101 | -0.0550 | False |

## Confidence Buckets

| bucket | count | mean_confidence | accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| low | 101 | 0.3700 | 0.4752 | -0.1052 | False |
| medium | 359 | 0.4901 | 0.4819 | 0.0082 | False |
| high | 303 | 0.7500 | 0.8383 | -0.0883 | False |

## Worst Misses

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| txg_decay_90d | Chelsea | Everton | A | H | 0.8590 | 0.0420 | 3.1701 |
| champion_dc_xg | Chelsea | Everton | A | H | 0.8531 | 0.0450 | 3.1021 |
| dc_fit_rho_each_fold | Chelsea | Everton | A | H | 0.8550 | 0.0460 | 3.0791 |
| txg_decay_45d | Chelsea | Everton | A | H | 0.8490 | 0.0470 | 3.0576 |
| txg_alpha_025 | Chelsea | Everton | A | H | 0.8470 | 0.0470 | 3.0576 |

## Best High-Confidence Correct

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| txg_decay_45d | Arsenal | Leicester City | H | H | 0.9280 | 0.9280 | 0.0747 |
| dc_fit_rho_each_fold | Arsenal | Leicester City | H | H | 0.9211 | 0.9211 | 0.0822 |
| champion_dc_xg | Arsenal | Leicester City | H | H | 0.9189 | 0.9189 | 0.0846 |
| txg_alpha_025 | Arsenal | Leicester City | H | H | 0.9140 | 0.9140 | 0.0899 |
| txg_xg_pseudocount_010 | Arsenal | Leicester City | H | H | 0.9110 | 0.9110 | 0.0932 |

## Favourite Breakdown

| predicted_outcome | confidence_bucket | n | accuracy | mean_confidence | mean_actual_probability | mean_log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| A | high | 120 | 0.7667 | 0.7304 | 0.6097 | 0.6282 |
| A | low | 40 | 0.3500 | 0.3721 | 0.3368 | 1.0941 |
| A | medium | 139 | 0.4892 | 0.4834 | 0.3620 | 1.0806 |
| D | low | 17 | 0.6471 | 0.3615 | 0.3499 | 1.0539 |
| H | high | 183 | 0.8852 | 0.7629 | 0.6898 | 0.5034 |
| H | low | 44 | 0.5227 | 0.3713 | 0.3414 | 1.0819 |
| H | medium | 220 | 0.4773 | 0.4944 | 0.3719 | 1.0644 |
