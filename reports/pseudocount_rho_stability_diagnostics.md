# Pseudocount and Rho Stability Diagnostics

## Executive Summary

Best by log loss: `dc_fit_rho_xg_pseudocount_025` (0.8617). Best by Brier: `dc_fit_rho_xg_pseudocount_025` (0.5013). Champion reference: log loss 0.8669, Brier 0.5052.

No candidate should be promoted from this branch alone.

## Inputs and Protocol

- CSV used: `data\exports\wsl_match_data_xg_phase3_20260626_fresh.csv`
- Evaluation window: 2025-10-01 to 2026-05-16
- Minimum training matches: 12
- Rolling folds: 19
- Train rows are strictly before each test window.
- xG columns required and verified: `home_np_xg`, `away_np_xg`, `home_xg`, `away_xg`.

## Aggregate Leaderboard

| rank | model_name | candidate_category | log_loss | brier_score | accuracy | average_max_probability | average_entropy | high_confidence_wrong_count | actual_probability_below_05_count | actual_probability_below_10_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | dc_fit_rho_xg_pseudocount_025 | combined_diagnostic | 0.8617 | 0.5013 | 0.6239 | 0.5783 | 0.9212 | 5 | 0 | 1 |
| 2 | dc_fit_rho_each_fold_wide_grid | rho_boundary_diagnostic | 0.8621 | 0.5025 | 0.6147 | 0.5876 | 0.9001 | 6 | 1 | 1 |
| 3 | dc_fit_rho_each_fold | rho_stability | 0.8623 | 0.5027 | 0.6147 | 0.5861 | 0.9010 | 6 | 1 | 1 |
| 4 | txg_xg_pseudocount_025 | pseudocount_stability | 0.8635 | 0.5024 | 0.6239 | 0.5701 | 0.9271 | 5 | 0 | 1 |
| 5 | txg_xg_pseudocount_020 | pseudocount_stability | 0.8636 | 0.5026 | 0.6147 | 0.5714 | 0.9232 | 5 | 0 | 1 |
| 6 | txg_xg_pseudocount_015 | pseudocount_stability | 0.8638 | 0.5030 | 0.6147 | 0.5729 | 0.9184 | 6 | 0 | 1 |
| 7 | txg_xg_pseudocount_010 | pseudocount_stability | 0.8649 | 0.5038 | 0.6330 | 0.5751 | 0.9123 | 6 | 1 | 1 |
| 8 | dc_rho_mild_minus_08 | rho_stability | 0.8650 | 0.5041 | 0.6330 | 0.5820 | 0.9031 | 6 | 1 | 1 |
| 9 | champion_dc_xg | champion_reference | 0.8669 | 0.5052 | 0.6239 | 0.5784 | 0.9046 | 6 | 1 | 1 |

## Pseudocount Stability

| xg_pseudocount | model_name | log_loss | brier_score | accuracy | average_max_probability | average_entropy | folds_better_log_loss_vs_005 | folds_worse_log_loss_vs_005 | folds_better_brier_vs_005 | folds_worse_brier_vs_005 | draw_recall | draw_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0500 | champion_dc_xg | 0.8669 | 0.5052 | 0.6239 | 0.5784 | 0.9046 | 0 | 0 | 0 | 0 | 0.0952 | 1.3243 |
| 0.1000 | txg_xg_pseudocount_010 | 0.8649 | 0.5038 | 0.6330 | 0.5751 | 0.9123 | 11 | 8 | 12 | 7 | 0.0952 | 1.3447 |
| 0.1500 | txg_xg_pseudocount_015 | 0.8638 | 0.5030 | 0.6147 | 0.5729 | 0.9184 | 11 | 8 | 12 | 7 | 0.0000 | 1.3640 |
| 0.2000 | txg_xg_pseudocount_020 | 0.8636 | 0.5026 | 0.6147 | 0.5714 | 0.9232 | 11 | 8 | 12 | 7 | 0.0000 | 1.3835 |
| 0.2500 | txg_xg_pseudocount_025 | 0.8635 | 0.5024 | 0.6239 | 0.5701 | 0.9271 | 11 | 8 | 12 | 7 | 0.0000 | 1.4019 |

Best pseudocount row: `txg_xg_pseudocount_025`. It beat the 0.05 reference on 11 folds by log loss and 12 folds by Brier. `0.25` continued the probability-quality improvement, so the saturation boundary did not reverse. However, `txg_xg_pseudocount_025` reduced draw recall from 0.0952 to 0.0000, so it is not the cleanest shadow candidate.

## Fitted-Rho Stability

| model_name | log_loss | brier_score | accuracy | rho_min | rho_max | rho_mean | rho_median | rho_std | grid_min_hits | grid_max_hits | folds_better_log_loss_vs_champion | folds_worse_log_loss_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| champion_dc_xg | 0.8669 | 0.5052 | 0.6239 | -0.1300 | -0.1300 | -0.1300 | -0.1300 | 0.0000 | 0 | 0 | 0 | 0 |
| dc_rho_mild_minus_08 | 0.8650 | 0.5041 | 0.6330 | -0.0800 | -0.0800 | -0.0800 | -0.0800 | 0.0000 | 0 | 0 | 11 | 8 |
| dc_fit_rho_each_fold | 0.8623 | 0.5027 | 0.6147 | -0.2400 | 0.0100 | -0.0389 | 0.0000 | 0.0738 | 0 | 8 | 12 | 7 |
| dc_fit_rho_each_fold_wide_grid | 0.8621 | 0.5025 | 0.6147 | -0.2400 | 0.1000 | -0.0200 | 0.0000 | 0.0905 | 0 | 2 | 12 | 7 |

Current fitted rho improved on 12 folds by log loss, with 8 max-boundary hits. The wide-grid diagnostic had 2 max-boundary hits and rho max 0.1000. Treat fitted rho as diagnostic unless boundary behavior is stable under future validation.

## Combined Diagnostic

| model_name | log_loss | brier_score | accuracy | average_max_probability | average_entropy | folds_better_log_loss_vs_champion | folds_better_brier_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dc_fit_rho_xg_pseudocount_025 | 0.8617 | 0.5013 | 0.6239 | 0.5783 | 0.9212 | 11 | 12 |

`dc_fit_rho_xg_pseudocount_025` was run as diagnostic only. It beat champion on 11 folds by log loss and 12 folds by Brier.

## Candidates Beating Champion

| model_name | candidate_category | log_loss | brier_score | accuracy |
| --- | --- | --- | --- | --- |
| dc_fit_rho_xg_pseudocount_025 | combined_diagnostic | 0.8617 | 0.5013 | 0.6239 |
| dc_fit_rho_each_fold_wide_grid | rho_boundary_diagnostic | 0.8621 | 0.5025 | 0.6147 |
| dc_fit_rho_each_fold | rho_stability | 0.8623 | 0.5027 | 0.6147 |
| txg_xg_pseudocount_025 | pseudocount_stability | 0.8635 | 0.5024 | 0.6239 |
| txg_xg_pseudocount_020 | pseudocount_stability | 0.8636 | 0.5026 | 0.6147 |
| txg_xg_pseudocount_015 | pseudocount_stability | 0.8638 | 0.5030 | 0.6147 |
| txg_xg_pseudocount_010 | pseudocount_stability | 0.8649 | 0.5038 | 0.6330 |
| dc_rho_mild_minus_08 | rho_stability | 0.8650 | 0.5041 | 0.6330 |

## Carry-Forward Recommendation

- Carry forward `txg_xg_pseudocount_010` as the safest pseudocount shadow hypothesis.
- Keep `txg_xg_pseudocount_025` as a probability-quality diagnostic because draw recall fell to zero.
- Keep fitted-rho rows diagnostic until boundary behavior is better understood.
- Keep `dc_fit_rho_xg_pseudocount_025` diagnostic-only; it stacks two uncertain levers.
- Park decay changes for this branch; they were not re-tested here.

## Guardrails

- The operational champion implementation remains frozen.
- The wider rho grid row is diagnostic only.
- The `0.25` pseudocount row is a saturation check, not a broad search.
- Market odds, neural networks, embeddings, and broad grids remain out of scope.
