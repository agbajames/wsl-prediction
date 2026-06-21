# Non-Market Blending Experiment

## First-Run Summary

This offline experiment keeps `champion_dc_xg` as the unchanged operational reference and tests
a small predeclared set of fixed-weight blends over existing non-market model probabilities.

- Champion reference: Brier 0.5052, log loss 0.8669, accuracy 0.6239.
- Best ranked row: `dc_fit_rho_each_fold` with Brier 0.5027, log loss 0.8623, accuracy 0.6147.
- Best blend: `blend_dc_fit_txg_50_50` with Brier 0.5031, log loss 0.8634, accuracy 0.6239.

No production promotion should happen from this PR alone; these blends are evaluation-only
signals for future investigation.

## Components

`champion_dc_xg`, `dc_fit_rho_each_fold`, `txg_xg_pseudocount_010`, `regularised_team_strength`, `improved_logistic_regression`, `random_forest`, `neural_network`

## Predeclared Blends

| model_name | components | weights | change |
| --- | --- | --- | --- |
| blend_champion_regularised_50_50 | champion_dc_xg, regularised_team_strength | 0.50, 0.50 | Even fixed blend of champion Dixon-Coles/xG and regularised team strength. |
| blend_champion_regularised_70_30 | champion_dc_xg, regularised_team_strength | 0.70, 0.30 | Champion-led blend with the strongest standalone statistical challenger. |
| blend_champion_improved_logistic_70_30 | champion_dc_xg, improved_logistic_regression | 0.70, 0.30 | Champion-led blend with the Phase 8B improved logistic challenger. |
| blend_champion_random_forest_70_30 | champion_dc_xg, random_forest | 0.70, 0.30 | Champion-led blend with the Phase 8B random forest challenger. |
| blend_champion_regularised_improved_logistic_60_20_20 | champion_dc_xg, regularised_team_strength, improved_logistic_regression | 0.60, 0.20, 0.20 | Three-model blend anchored on champion with two non-market challengers. |
| blend_dc_fit_txg_50_50 | dc_fit_rho_each_fold, txg_xg_pseudocount_010 | 0.50, 0.50 | Small Phase 8A champion-family reference blend of fitted-rho and conservative xG shrinkage. |

## Model Comparison

| rank | model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | dc_fit_rho_each_fold | 109 | 0.5027 | 0.8623 | 0.6147 |
| 2 | blend_dc_fit_txg_50_50 | 109 | 0.5031 | 0.8634 | 0.6239 |
| 3 | txg_xg_pseudocount_010 | 109 | 0.5038 | 0.8649 | 0.6330 |
| 4 | champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 |
| 5 | blend_champion_regularised_70_30 | 109 | 0.5072 | 0.8713 | 0.6055 |
| 6 | blend_champion_improved_logistic_70_30 | 109 | 0.5068 | 0.8720 | 0.5963 |
| 7 | blend_champion_regularised_improved_logistic_60_20_20 | 109 | 0.5075 | 0.8729 | 0.6055 |
| 8 | blend_champion_random_forest_70_30 | 109 | 0.5095 | 0.8735 | 0.6055 |
| 9 | blend_champion_regularised_50_50 | 109 | 0.5105 | 0.8771 | 0.6055 |
| 10 | regularised_team_strength | 109 | 0.5256 | 0.9001 | 0.5963 |
| 11 | improved_logistic_regression | 109 | 0.5384 | 0.9356 | 0.5780 |
| 12 | random_forest | 109 | 0.5491 | 0.9385 | 0.5872 |
| 13 | neural_network | 109 | 0.5567 | 0.9762 | 0.6055 |

## Calibration

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 162 | 0.3758 | 0.4136 | -0.0378 | False |
| 40%-60% | 700 | 0.4947 | 0.4786 | 0.0161 | False |
| 60%-80% | 429 | 0.6928 | 0.7995 | -0.1067 | False |
| 80%-100% | 126 | 0.8517 | 0.9048 | -0.0531 | False |

## Confidence Buckets

| bucket | count | mean_confidence | accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| low | 162 | 0.3758 | 0.4136 | -0.0378 | False |
| medium | 700 | 0.4947 | 0.4786 | 0.0161 | False |
| high | 555 | 0.7289 | 0.8234 | -0.0945 | False |

## Worst Misses

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neural_network | Chelsea | Everton | A | H | 0.8987 | 0.0218 | 3.8269 |
| neural_network | Tottenham Hotspur | Brighton | H | A | 0.8960 | 0.0334 | 3.3992 |
| neural_network | Aston Villa | Leicester City | D | H | 0.8914 | 0.0338 | 3.3880 |
| champion_dc_xg | Chelsea | Everton | A | H | 0.8531 | 0.0450 | 3.1021 |
| dc_fit_rho_each_fold | Chelsea | Everton | A | H | 0.8550 | 0.0460 | 3.0791 |

## Best High-Confidence Correct

| model_name | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neural_network | London City Lionesses | Liverpool | H | H | 0.9595 | 0.9595 | 0.0413 |
| neural_network | Manchester City | West Ham United | H | H | 0.9350 | 0.9350 | 0.0672 |
| neural_network | Arsenal | Liverpool | H | H | 0.9312 | 0.9312 | 0.0713 |
| dc_fit_rho_each_fold | Arsenal | Leicester City | H | H | 0.9211 | 0.9211 | 0.0822 |
| champion_dc_xg | Arsenal | Leicester City | H | H | 0.9189 | 0.9189 | 0.0846 |

## Favourite Breakdown

| predicted_outcome | confidence_bucket | n | accuracy | mean_confidence | mean_actual_probability | mean_log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| A | high | 214 | 0.7477 | 0.7095 | 0.5739 | 0.7261 |
| A | low | 64 | 0.3594 | 0.3773 | 0.3377 | 1.0934 |
| A | medium | 279 | 0.4695 | 0.4860 | 0.3614 | 1.0840 |
| D | low | 11 | 0.5455 | 0.3654 | 0.3443 | 1.0730 |
| D | medium | 1 | 0.0000 | 0.4180 | 0.4128 | 0.8849 |
| H | high | 341 | 0.8710 | 0.7410 | 0.6660 | 0.5395 |
| H | low | 87 | 0.4368 | 0.3761 | 0.3413 | 1.0820 |
| H | medium | 420 | 0.4857 | 0.5006 | 0.3754 | 1.0548 |
