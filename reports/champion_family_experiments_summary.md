# Champion-Family Calibration and Decay Experiments

## Executive Summary

Best candidate by log loss: `additive_draw_-0.050` (log loss 0.8581, Brier 0.5011, accuracy 0.6147).
Best candidate by Brier score: `additive_draw_-0.050` (Brier 0.5011, log loss 0.8581, accuracy 0.6147).
Champion reference `champion_dc_xg`: log loss 0.8669, Brier 0.5052, accuracy 0.6239.

No variant should be promoted from this one offline window alone. Candidates that beat the champion
remain shadow-ready hypotheses pending future-fixture validation.

## Inputs and Protocol

- CSV used: `data/exports/wsl_match_data_xg_phase3_20260626_fresh.csv`
- Evaluation window: 2025-10-01 to 2026-05-16
- Minimum training matches: 12
- Test window days: 7
- Step days: 7
- Rolling folds: 19
- Evaluated fixtures per full candidate: 109
- xG columns required and verified: `home_np_xg`, `away_np_xg`, `home_xg`, `away_xg`.

## Candidate List

| candidate_group | model_name | description |
| --- | --- | --- |
| core_champion_family | champion_dc_xg | Frozen champion reference: 60-day decay, fixed rho -0.13, xG pseudocount 0.05. |
| core_champion_family | dc_rho_mild_minus_08 | Less negative fixed Dixon-Coles rho (-0.08). |
| core_champion_family | dc_fit_rho_each_fold | Fit rho inside each training fold using the existing champion grid search. |
| core_champion_family | txg_xg_pseudocount_010 | Conservative np_xG floor/shrinkage: xG pseudocount 0.10. |
| xg_pseudocount_sensitivity | txg_xg_pseudocount_015 | Pseudocount sensitivity: xG pseudocount 0.15. |
| xg_pseudocount_sensitivity | txg_xg_pseudocount_020 | Pseudocount sensitivity upper bound: xG pseudocount 0.20. |
| time_decay_sensitivity | txg_decay_75d | Conservative recency half-life sensitivity: 75 days. |
| time_decay_sensitivity | txg_decay_90d | Longer recency half-life diagnostic: 90 days. |
| time_decay_sensitivity | txg_decay_45d | Shorter recency half-life diagnostic holdover: 45 days. |
| draw_calibration_diagnostic | additive_draw_-0.025 | additive draw adjustment (-0.025). |
| draw_calibration_diagnostic | multiplicative_draw_0.95 | multiplicative draw adjustment (0.95). |
| draw_calibration_diagnostic | additive_draw_-0.050 | additive draw adjustment (-0.05). |
| fixed_champion_family_blend | blend_dc_fit_txg_50_50 | Fixed blend of fitted-rho and xG-pseudocount champion-family candidates. |
| fixed_champion_family_blend | blend_champion_dc_fit_50_50 | Fixed blend of frozen champion and fitted-rho candidate. |
| fixed_champion_family_blend | blend_champion_txg_50_50 | Fixed blend of frozen champion and xG-pseudocount candidate. |
| fixed_champion_family_blend | blend_champion_dc_fit_txg_34_33_33 | Fixed near-equal blend of the three champion-family references. |

## Aggregate Leaderboard

| rank | model_name | candidate_category | n_matches | n_folds | log_loss | brier_score | accuracy | average_max_probability | high_confidence_wrong_count | actual_probability_below_05_count | actual_probability_below_10_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | additive_draw_-0.050 | draw_calibration_diagnostic | 109 | 19 | 0.8581 | 0.5011 | 0.6147 | 0.6155 | 2 | 1 | 3 |
| 2 | additive_draw_-0.025 | draw_calibration_diagnostic | 109 | 19 | 0.8607 | 0.5021 | 0.6147 | 0.5967 | 2 | 1 | 1 |
| 3 | dc_fit_rho_each_fold | core_champion_family | 109 | 19 | 0.8623 | 0.5027 | 0.6147 | 0.5861 | 1 | 1 | 1 |
| 4 | blend_dc_fit_txg_50_50 | fixed_champion_family_blend | 109 | 19 | 0.8634 | 0.5031 | 0.6239 | 0.5805 | 1 | 1 | 1 |
| 5 | txg_xg_pseudocount_020 | xg_pseudocount_sensitivity | 109 | 19 | 0.8636 | 0.5026 | 0.6147 | 0.5714 | 1 | 0 | 1 |
| 6 | txg_xg_pseudocount_015 | xg_pseudocount_sensitivity | 109 | 19 | 0.8638 | 0.5030 | 0.6147 | 0.5729 | 1 | 0 | 1 |
| 7 | blend_champion_dc_fit_50_50 | fixed_champion_family_blend | 109 | 19 | 0.8644 | 0.5038 | 0.6239 | 0.5821 | 1 | 1 | 1 |
| 8 | blend_champion_dc_fit_txg_34_33_33 | fixed_champion_family_blend | 109 | 19 | 0.8645 | 0.5038 | 0.6330 | 0.5797 | 1 | 1 | 1 |
| 9 | multiplicative_draw_0.95 | draw_calibration_diagnostic | 109 | 19 | 0.8645 | 0.5039 | 0.6330 | 0.5846 | 1 | 1 | 1 |
| 10 | txg_xg_pseudocount_010 | core_champion_family | 109 | 19 | 0.8649 | 0.5038 | 0.6330 | 0.5751 | 1 | 1 | 1 |
| 11 | dc_rho_mild_minus_08 | core_champion_family | 109 | 19 | 0.8650 | 0.5041 | 0.6330 | 0.5820 | 1 | 1 | 1 |
| 12 | blend_champion_txg_50_50 | fixed_champion_family_blend | 109 | 19 | 0.8658 | 0.5045 | 0.6330 | 0.5767 | 1 | 1 | 1 |
| 13 | txg_decay_75d | time_decay_sensitivity | 109 | 19 | 0.8668 | 0.5052 | 0.6147 | 0.5788 | 1 | 1 | 1 |
| 14 | champion_dc_xg | core_champion_family | 109 | 19 | 0.8669 | 0.5052 | 0.6239 | 0.5784 | 1 | 1 | 1 |
| 15 | txg_decay_90d | time_decay_sensitivity | 109 | 19 | 0.8672 | 0.5055 | 0.6147 | 0.5793 | 1 | 1 | 1 |
| 16 | txg_decay_45d | time_decay_sensitivity | 109 | 19 | 0.8684 | 0.5061 | 0.6239 | 0.5780 | 2 | 1 | 1 |

## Fold-Level Observations

| model_name | folds_better_log_loss | folds_better_brier | mean_log_loss_delta_vs_champion | mean_brier_delta_vs_champion |
| --- | --- | --- | --- | --- |
| additive_draw_-0.025 | 10 | 11 | -0.0049 | -0.0024 |
| additive_draw_-0.050 | 10 | 11 | -0.0061 | -0.0027 |
| blend_champion_dc_fit_50_50 | 12 | 12 | -0.0024 | -0.0014 |
| blend_champion_dc_fit_txg_34_33_33 | 12 | 13 | -0.0023 | -0.0015 |
| blend_champion_txg_50_50 | 11 | 12 | -0.0010 | -0.0008 |
| blend_dc_fit_txg_50_50 | 12 | 12 | -0.0033 | -0.0021 |
| dc_fit_rho_each_fold | 12 | 12 | -0.0044 | -0.0025 |
| dc_rho_mild_minus_08 | 11 | 12 | -0.0017 | -0.0010 |
| multiplicative_draw_0.95 | 12 | 11 | -0.0020 | -0.0011 |
| txg_decay_45d | 8 | 6 | 0.0017 | 0.0014 |
| txg_decay_75d | 11 | 13 | -0.0001 | -0.0002 |
| txg_decay_90d | 10 | 12 | 0.0003 | 0.0001 |
| txg_xg_pseudocount_010 | 11 | 12 | -0.0019 | -0.0015 |
| txg_xg_pseudocount_015 | 11 | 12 | -0.0030 | -0.0025 |
| txg_xg_pseudocount_020 | 11 | 12 | -0.0031 | -0.0030 |

## Fitted-Rho Diagnostics

- Fitted-rho fold count: 19
- Rho range: -0.2400 to 0.0100
- Rho mean/median/std: -0.0389 / 0.0000 / 0.0738
- Grid boundary hits: min=0, max=8
- Folds improved vs champion: log loss=12, Brier=12

| model_name | fold_id | train_size | test_size | resolved_rho | rho_hits_grid_min | rho_hits_grid_max | log_loss_delta_vs_champion | brier_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| champion_dc_xg | fold_001 | 23 | 6 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_002 | 29 | 6 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_003 | 35 | 6 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_004 | 41 | 6 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_005 | 47 | 6 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_006 | 53 | 6 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_007 | 59 | 7 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_008 | 66 | 6 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_009 | 72 | 6 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_010 | 78 | 5 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_011 | 83 | 6 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_012 | 89 | 5 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_013 | 94 | 4 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_014 | 98 | 8 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_015 | 106 | 6 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_016 | 112 | 5 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_017 | 117 | 6 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_018 | 123 | 2 | -0.1300 | False | False | 0.0000 | 0.0000 |
| champion_dc_xg | fold_019 | 125 | 7 | -0.1300 | False | False | 0.0000 | 0.0000 |
| dc_rho_mild_minus_08 | fold_001 | 23 | 6 | -0.0800 | False | False | -0.0011 | -0.0005 |
| dc_rho_mild_minus_08 | fold_002 | 29 | 6 | -0.0800 | False | False | 0.0011 | 0.0003 |
| dc_rho_mild_minus_08 | fold_003 | 35 | 6 | -0.0800 | False | False | 0.0012 | -0.0014 |
| dc_rho_mild_minus_08 | fold_004 | 41 | 6 | -0.0800 | False | False | 0.0118 | 0.0083 |
| dc_rho_mild_minus_08 | fold_005 | 47 | 6 | -0.0800 | False | False | 0.0034 | -0.0008 |
| dc_rho_mild_minus_08 | fold_006 | 53 | 6 | -0.0800 | False | False | -0.0119 | -0.0045 |
| dc_rho_mild_minus_08 | fold_007 | 59 | 7 | -0.0800 | False | False | 0.0013 | 0.0019 |
| dc_rho_mild_minus_08 | fold_008 | 66 | 6 | -0.0800 | False | False | 0.0064 | 0.0030 |
| dc_rho_mild_minus_08 | fold_009 | 72 | 6 | -0.0800 | False | False | -0.0113 | -0.0076 |
| dc_rho_mild_minus_08 | fold_010 | 78 | 5 | -0.0800 | False | False | -0.0121 | -0.0068 |
| dc_rho_mild_minus_08 | fold_011 | 83 | 6 | -0.0800 | False | False | -0.0161 | -0.0077 |
| dc_rho_mild_minus_08 | fold_012 | 89 | 5 | -0.0800 | False | False | -0.0080 | -0.0048 |
| dc_rho_mild_minus_08 | fold_013 | 94 | 4 | -0.0800 | False | False | -0.0026 | -0.0032 |
| dc_rho_mild_minus_08 | fold_014 | 98 | 8 | -0.0800 | False | False | 0.0098 | 0.0056 |
| dc_rho_mild_minus_08 | fold_015 | 106 | 6 | -0.0800 | False | False | -0.0037 | -0.0023 |
| dc_rho_mild_minus_08 | fold_016 | 112 | 5 | -0.0800 | False | False | -0.0051 | -0.0014 |
| dc_rho_mild_minus_08 | fold_017 | 117 | 6 | -0.0800 | False | False | -0.0026 | 0.0004 |
| dc_rho_mild_minus_08 | fold_018 | 123 | 2 | -0.0800 | False | False | 0.0141 | 0.0073 |
| dc_rho_mild_minus_08 | fold_019 | 125 | 7 | -0.0800 | False | False | -0.0074 | -0.0051 |
| dc_fit_rho_each_fold | fold_001 | 23 | 6 | -0.2100 | False | False | 0.0017 | 0.0007 |
| dc_fit_rho_each_fold | fold_002 | 29 | 6 | -0.2400 | False | False | -0.0025 | -0.0004 |
| dc_fit_rho_each_fold | fold_003 | 35 | 6 | -0.1200 | False | False | -0.0003 | -0.0006 |
| dc_fit_rho_each_fold | fold_004 | 41 | 6 | -0.0700 | False | False | 0.0142 | 0.0101 |
| dc_fit_rho_each_fold | fold_005 | 47 | 6 | -0.0300 | False | False | 0.0093 | -0.0005 |
| dc_fit_rho_each_fold | fold_006 | 53 | 6 | -0.0900 | False | False | -0.0108 | -0.0041 |
| dc_fit_rho_each_fold | fold_007 | 59 | 7 | 0.0100 | False | True | 0.0071 | 0.0069 |
| dc_fit_rho_each_fold | fold_008 | 66 | 6 | 0.0100 | False | True | 0.0197 | 0.0094 |
| dc_fit_rho_each_fold | fold_009 | 72 | 6 | 0.0100 | False | True | -0.0317 | -0.0211 |
| dc_fit_rho_each_fold | fold_010 | 78 | 5 | 0.0100 | False | True | -0.0340 | -0.0190 |
| dc_fit_rho_each_fold | fold_011 | 83 | 6 | 0.0100 | False | True | -0.0455 | -0.0219 |
| dc_fit_rho_each_fold | fold_012 | 89 | 5 | 0.0100 | False | True | -0.0215 | -0.0125 |
| dc_fit_rho_each_fold | fold_013 | 94 | 4 | 0.0100 | False | True | -0.0059 | -0.0079 |
| dc_fit_rho_each_fold | fold_014 | 98 | 8 | 0.0100 | False | True | 0.0282 | 0.0157 |
| dc_fit_rho_each_fold | fold_015 | 106 | 6 | 0.0000 | False | False | -0.0090 | -0.0054 |
| dc_fit_rho_each_fold | fold_016 | 112 | 5 | -0.0100 | False | False | -0.0135 | -0.0041 |
| dc_fit_rho_each_fold | fold_017 | 117 | 6 | 0.0000 | False | False | -0.0065 | 0.0011 |
| dc_fit_rho_each_fold | fold_018 | 123 | 2 | -0.0200 | False | False | 0.0320 | 0.0170 |
| dc_fit_rho_each_fold | fold_019 | 125 | 7 | -0.0300 | False | False | -0.0148 | -0.0101 |

## Pseudocount Sensitivity

| model_name | log_loss | brier_score | accuracy | average_max_probability |
| --- | --- | --- | --- | --- |
| txg_xg_pseudocount_020 | 0.8636 | 0.5026 | 0.6147 | 0.5714 |
| txg_xg_pseudocount_015 | 0.8638 | 0.5030 | 0.6147 | 0.5729 |
| txg_xg_pseudocount_010 | 0.8649 | 0.5038 | 0.6330 | 0.5751 |
| champion_dc_xg | 0.8669 | 0.5052 | 0.6239 | 0.5784 |

## Decay Sensitivity

| model_name | log_loss | brier_score | accuracy | average_max_probability |
| --- | --- | --- | --- | --- |
| txg_decay_75d | 0.8668 | 0.5052 | 0.6147 | 0.5788 |
| champion_dc_xg | 0.8669 | 0.5052 | 0.6239 | 0.5784 |
| txg_decay_90d | 0.8672 | 0.5055 | 0.6147 | 0.5793 |
| txg_decay_45d | 0.8684 | 0.5061 | 0.6239 | 0.5780 |

## Draw Calibration Diagnostics

| model_name | log_loss | brier_score | accuracy | actual_draws | draw_prediction_rate | avg_predicted_draw_probability | draw_recall | draw_log_loss | non_draw_log_loss | non_draw_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| additive_draw_-0.050 | 0.8581 | 0.5011 | 0.6147 | 21 | 0.0000 | 0.1931 | 0.0000 | 1.5464 | 0.6939 | 0.7614 |
| additive_draw_-0.025 | 0.8607 | 0.5021 | 0.6147 | 21 | 0.0000 | 0.2181 | 0.0000 | 1.4283 | 0.7253 | 0.7614 |
| multiplicative_draw_0.95 | 0.8645 | 0.5039 | 0.6330 | 21 | 0.0183 | 0.2341 | 0.0952 | 1.3617 | 0.7459 | 0.7614 |
| champion_dc_xg | 0.8669 | 0.5052 | 0.6239 | 21 | 0.0367 | 0.2431 | 0.0952 | 1.3243 | 0.7578 | 0.7500 |

## Fixed Blend Results

| model_name | log_loss | brier_score | accuracy | average_max_probability |
| --- | --- | --- | --- | --- |
| blend_dc_fit_txg_50_50 | 0.8634 | 0.5031 | 0.6239 | 0.5805 |
| blend_champion_dc_fit_50_50 | 0.8644 | 0.5038 | 0.6239 | 0.5821 |
| blend_champion_dc_fit_txg_34_33_33 | 0.8645 | 0.5038 | 0.6330 | 0.5797 |
| blend_champion_txg_50_50 | 0.8658 | 0.5045 | 0.6330 | 0.5767 |

## Comparison Against Champion

| model_name | candidate_category | log_loss | brier_score | accuracy |
| --- | --- | --- | --- | --- |
| additive_draw_-0.050 | draw_calibration_diagnostic | 0.8581 | 0.5011 | 0.6147 |
| additive_draw_-0.025 | draw_calibration_diagnostic | 0.8607 | 0.5021 | 0.6147 |
| dc_fit_rho_each_fold | core_champion_family | 0.8623 | 0.5027 | 0.6147 |
| blend_dc_fit_txg_50_50 | fixed_champion_family_blend | 0.8634 | 0.5031 | 0.6239 |
| txg_xg_pseudocount_020 | xg_pseudocount_sensitivity | 0.8636 | 0.5026 | 0.6147 |
| txg_xg_pseudocount_015 | xg_pseudocount_sensitivity | 0.8638 | 0.5030 | 0.6147 |
| blend_champion_dc_fit_50_50 | fixed_champion_family_blend | 0.8644 | 0.5038 | 0.6239 |
| blend_champion_dc_fit_txg_34_33_33 | fixed_champion_family_blend | 0.8645 | 0.5038 | 0.6330 |
| multiplicative_draw_0.95 | draw_calibration_diagnostic | 0.8645 | 0.5039 | 0.6330 |
| txg_xg_pseudocount_010 | core_champion_family | 0.8649 | 0.5038 | 0.6330 |
| dc_rho_mild_minus_08 | core_champion_family | 0.8650 | 0.5041 | 0.6330 |
| blend_champion_txg_50_50 | fixed_champion_family_blend | 0.8658 | 0.5045 | 0.6330 |
| txg_decay_75d | time_decay_sensitivity | 0.8668 | 0.5052 | 0.6147 |

## Recommendation

- `additive_draw_-0.050` leads aggregate log loss/Brier in this offline run.
- Carry forward candidates that beat the champion on log loss or Brier as shadow-ready hypotheses only.
- `additive_draw_-0.050` also reduced draw recall to zero, so keep it as a draw-calibration diagnostic rather than a promotion-ready candidate.
- Fitted rho hit a grid boundary in at least one fold, so treat rho fitting as potentially noisy.
- Prioritise future validation over additional offline tuning.

## Guardrails

- `champion_dc_xg` remains the frozen operational/reference model.
- These candidates are separate offline variants, post-processing diagnostics, or fixed blends.
- Market odds are not used as features or training inputs.
- Neural-network work remains parked.
- Do not promote a candidate from this one offline window alone.
