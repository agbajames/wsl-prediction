# Dixon-Coles Configuration Variant Experiment

## First-Run Summary

This offline experiment keeps `champion_dc_xg` as the unchanged reference and tests a small
predeclared grid of champion-family configuration variants.

Best-ranked variant: `dc_fit_rho_each_fold` versus champion reference `champion_dc_xg`.

- Champion reference: Brier 0.5052, log loss 0.8669, accuracy 0.6239.
- Best ranked row: `dc_fit_rho_each_fold` with Brier 0.5027, log loss 0.8623, accuracy 0.6147.

## Predeclared Variants

| model_name | change |
| --- | --- |
| champion_dc_xg | Frozen champion reference configuration. |
| dc_rho_mild_minus_08 | Less negative fixed Dixon-Coles rho (-0.08). |
| dc_rho_stronger_minus_18 | More negative fixed Dixon-Coles rho (-0.18). |
| dc_fit_rho_each_fold | Fit rho inside each training fold using the existing champion grid search. |
| dc_score_grid_10 | Extend the scoreline truncation grid from 8 to 10 goals. |
| dc_alpha_030 | Increase xG strength ridge regularisation from 0.15 to 0.30. |
| dc_decay_30d | Use a shorter 30-day time-decay half-life. |
| dc_conservative_xg_shrinkage | Use larger np_xG pseudocount and stronger penalty-rate shrinkage. |

## Model Comparison

| rank | model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | dc_fit_rho_each_fold | 109 | 0.5027 | 0.8623 | 0.6147 |
| 2 | dc_rho_mild_minus_08 | 109 | 0.5041 | 0.8650 | 0.6330 |
| 3 | dc_conservative_xg_shrinkage | 109 | 0.5052 | 0.8669 | 0.6239 |
| 4 | champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 |
| 5 | dc_score_grid_10 | 109 | 0.5052 | 0.8670 | 0.6239 |
| 6 | dc_alpha_030 | 109 | 0.5058 | 0.8681 | 0.6147 |
| 7 | dc_rho_stronger_minus_18 | 109 | 0.5064 | 0.8691 | 0.6147 |
| 8 | dc_decay_30d | 109 | 0.5110 | 0.8757 | 0.6239 |

## Calibration

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 120 | 0.3719 | 0.4833 | -0.1114 | False |
| 40%-60% | 405 | 0.4906 | 0.4790 | 0.0116 | False |
| 60%-80% | 243 | 0.7072 | 0.8025 | -0.0953 | False |
| 80%-100% | 104 | 0.8566 | 0.9135 | -0.0569 | False |

## Confidence Buckets

| bucket | count | mean_confidence | accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| low | 120 | 0.3719 | 0.4833 | -0.1114 | False |
| medium | 404 | 0.4903 | 0.4777 | 0.0126 | False |
| high | 348 | 0.7516 | 0.8362 | -0.0846 | False |

## Worst Misses

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dc_rho_stronger_minus_18 | Chelsea | Everton | A | H | 0.8520 | 0.0430 | 3.1466 |
| dc_score_grid_10 | Chelsea | Everton | A | H | 0.8550 | 0.0440 | 3.1236 |
| champion_dc_xg | Chelsea | Everton | A | H | 0.8531 | 0.0450 | 3.1021 |
| dc_rho_mild_minus_08 | Chelsea | Everton | A | H | 0.8560 | 0.0460 | 3.0791 |
| dc_fit_rho_each_fold | Chelsea | Everton | A | H | 0.8550 | 0.0460 | 3.0791 |

## Best High-Confidence Correct

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dc_decay_30d | Arsenal | Leicester City | H | H | 0.9390 | 0.9390 | 0.0629 |
| dc_fit_rho_each_fold | Arsenal | Leicester City | H | H | 0.9211 | 0.9211 | 0.0822 |
| dc_rho_mild_minus_08 | Arsenal | Leicester City | H | H | 0.9191 | 0.9191 | 0.0844 |
| dc_score_grid_10 | Arsenal | Leicester City | H | H | 0.9190 | 0.9190 | 0.0845 |
| champion_dc_xg | Arsenal | Leicester City | H | H | 0.9189 | 0.9189 | 0.0846 |

## Favourite Breakdown

| predicted_outcome | confidence_bucket | n | accuracy | mean_confidence | mean_actual_probability | mean_log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| A | high | 138 | 0.7681 | 0.7318 | 0.6104 | 0.6280 |
| A | low | 50 | 0.4000 | 0.3754 | 0.3416 | 1.0797 |
| A | medium | 151 | 0.4834 | 0.4846 | 0.3598 | 1.0886 |
| D | low | 25 | 0.5600 | 0.3621 | 0.3476 | 1.0611 |
| H | high | 210 | 0.8810 | 0.7646 | 0.6891 | 0.5082 |
| H | low | 45 | 0.5333 | 0.3733 | 0.3405 | 1.0851 |
| H | medium | 253 | 0.4743 | 0.4937 | 0.3719 | 1.0650 |
