# Champion Favourite-Shrinkage Experiment

## Executive Summary

This offline experiment tests favourite shrinkage for `champion_dc_xg` on 109 evaluated matches.
Best variant by log loss: `original_champion`. Deltas versus original: Brier +0.0000, log loss +0.0000, accuracy +0.0000.
Recommendation: keep the original champion unchanged; no tested favourite shrinkage improved log loss.

## Methodology

- Apply fixed threshold and soft-cap shrinkage to high-confidence favourites.
- Thresholds tested: 0.65, 0.7, 0.75, 0.8
- Threshold shrink strengths tested: 0.05, 0.1, 0.15
- Soft caps tested: 0.7, 0.75, 0.8
- Soft-cap strengths tested: 0.25, 0.5, 0.75
- The underlying champion model, API, dashboard, and production outputs are unchanged.

## Original Champion Favourite Behaviour

| high_confidence_threshold | high_confidence_favourites | high_confidence_favourite_accuracy | mean_favourite_confidence | high_confidence_miss_count | high_confidence_correct_count | high_confidence_miss_log_loss | high_confidence_correct_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6500 | 38 | 0.8421 | 0.7706 | 6 | 32 | 2.0256 | 0.2602 |

## Favourite-Shrinkage Variants Tested

| variant_name | method | threshold | cap | strength | brier_score | log_loss | accuracy | high_confidence_favourites | high_confidence_miss_count | high_confidence_correct_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_champion | original |  |  | 0.0000 | 0.5052 | 0.8669 | 0.6239 | 38 | 6 | 32 |
| threshold_0.65_shrink_0.05 | threshold | 0.6500 |  | 0.0500 | 0.5104 | 0.8757 | 0.6239 | 38 | 6 | 32 |
| threshold_0.65_shrink_0.10 | threshold | 0.6500 |  | 0.1000 | 0.5147 | 0.8827 | 0.6239 | 38 | 6 | 32 |
| threshold_0.65_shrink_0.15 | threshold | 0.6500 |  | 0.1500 | 0.5186 | 0.8886 | 0.6239 | 38 | 6 | 32 |
| threshold_0.70_shrink_0.05 | threshold | 0.7000 |  | 0.0500 | 0.5076 | 0.8715 | 0.6239 | 38 | 6 | 32 |
| threshold_0.70_shrink_0.10 | threshold | 0.7000 |  | 0.1000 | 0.5103 | 0.8762 | 0.6239 | 38 | 6 | 32 |
| threshold_0.70_shrink_0.15 | threshold | 0.7000 |  | 0.1500 | 0.5131 | 0.8804 | 0.6239 | 38 | 6 | 32 |
| threshold_0.75_shrink_0.05 | threshold | 0.7500 |  | 0.0500 | 0.5067 | 0.8701 | 0.6239 | 38 | 6 | 32 |
| threshold_0.75_shrink_0.10 | threshold | 0.7500 |  | 0.1000 | 0.5088 | 0.8735 | 0.6239 | 38 | 6 | 32 |
| threshold_0.75_shrink_0.15 | threshold | 0.7500 |  | 0.1500 | 0.5100 | 0.8756 | 0.6239 | 38 | 6 | 32 |
| threshold_0.80_shrink_0.05 | threshold | 0.8000 |  | 0.0500 | 0.5065 | 0.8694 | 0.6239 | 38 | 6 | 32 |
| threshold_0.80_shrink_0.10 | threshold | 0.8000 |  | 0.1000 | 0.5074 | 0.8714 | 0.6239 | 38 | 6 | 32 |
| threshold_0.80_shrink_0.15 | threshold | 0.8000 |  | 0.1500 | 0.5076 | 0.8717 | 0.6239 | 38 | 6 | 32 |
| soft_cap_0.70_strength_0.25 | soft_cap |  | 0.7000 | 0.2500 | 0.5066 | 0.8698 | 0.6239 | 38 | 6 | 32 |
| soft_cap_0.70_strength_0.50 | soft_cap |  | 0.7000 | 0.5000 | 0.5086 | 0.8735 | 0.6239 | 38 | 6 | 32 |
| soft_cap_0.70_strength_0.75 | soft_cap |  | 0.7000 | 0.7500 | 0.5114 | 0.8779 | 0.6239 | 38 | 6 | 32 |
| soft_cap_0.75_strength_0.25 | soft_cap |  | 0.7500 | 0.2500 | 0.5060 | 0.8687 | 0.6239 | 38 | 6 | 32 |
| soft_cap_0.75_strength_0.50 | soft_cap |  | 0.7500 | 0.5000 | 0.5071 | 0.8708 | 0.6239 | 38 | 6 | 32 |
| soft_cap_0.75_strength_0.75 | soft_cap |  | 0.7500 | 0.7500 | 0.5085 | 0.8732 | 0.6239 | 38 | 6 | 32 |
| soft_cap_0.80_strength_0.25 | soft_cap |  | 0.8000 | 0.2500 | 0.5056 | 0.8679 | 0.6239 | 38 | 6 | 32 |
| soft_cap_0.80_strength_0.50 | soft_cap |  | 0.8000 | 0.5000 | 0.5062 | 0.8691 | 0.6239 | 38 | 6 | 32 |
| soft_cap_0.80_strength_0.75 | soft_cap |  | 0.8000 | 0.7500 | 0.5068 | 0.8703 | 0.6239 | 38 | 6 | 32 |

## Original Champion Vs Best Shrinkage Variant

| variant_name | brier_score | log_loss | accuracy | high_confidence_favourites | high_confidence_favourite_accuracy | mean_favourite_confidence | high_confidence_miss_count | high_confidence_correct_count | high_confidence_miss_log_loss | high_confidence_correct_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_champion | 0.5052 | 0.8669 | 0.6239 | 38 | 0.8421 | 0.7706 | 6 | 32 | 2.0256 | 0.2602 |
| original_champion | 0.5052 | 0.8669 | 0.6239 | 38 | 0.8421 | 0.7706 | 6 | 32 | 2.0256 | 0.2602 |

## Overall Metrics

| variant_name | brier_score | log_loss | accuracy | high_confidence_miss_count | high_confidence_correct_count | mean_favourite_confidence |
| --- | --- | --- | --- | --- | --- | --- |
| original_champion | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 |

## High-Confidence Favourite Metrics

| variant_name | high_confidence_threshold | high_confidence_favourites | high_confidence_favourite_accuracy | mean_favourite_confidence | high_confidence_miss_count | high_confidence_correct_count | high_confidence_miss_log_loss | high_confidence_correct_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_champion | 0.6500 | 38 | 0.8421 | 0.7706 | 6 | 32 | 2.0256 | 0.2602 |
| threshold_0.65_shrink_0.05 | 0.6500 | 38 | 0.8421 | 0.7248 | 6 | 32 | 1.8548 | 0.3222 |
| threshold_0.65_shrink_0.10 | 0.6500 | 38 | 0.8421 | 0.6910 | 6 | 32 | 1.7353 | 0.3685 |
| threshold_0.65_shrink_0.15 | 0.6500 | 38 | 0.8421 | 0.6697 | 6 | 32 | 1.6785 | 0.3992 |
| threshold_0.70_shrink_0.05 | 0.6500 | 38 | 0.8421 | 0.7368 | 6 | 32 | 1.8790 | 0.3035 |
| threshold_0.70_shrink_0.10 | 0.6500 | 38 | 0.8421 | 0.7155 | 6 | 32 | 1.8105 | 0.3322 |
| threshold_0.70_shrink_0.15 | 0.6500 | 38 | 0.8421 | 0.7016 | 6 | 32 | 1.7798 | 0.3522 |
| threshold_0.75_shrink_0.05 | 0.6500 | 38 | 0.8421 | 0.7493 | 6 | 32 | 1.9394 | 0.2872 |
| threshold_0.75_shrink_0.10 | 0.6500 | 38 | 0.8421 | 0.7354 | 6 | 32 | 1.9017 | 0.3059 |
| threshold_0.75_shrink_0.15 | 0.6500 | 38 | 0.8421 | 0.7303 | 6 | 32 | 1.8996 | 0.3137 |
| threshold_0.80_shrink_0.05 | 0.6500 | 38 | 0.8421 | 0.7567 | 6 | 32 | 1.9768 | 0.2779 |
| threshold_0.80_shrink_0.10 | 0.6500 | 38 | 0.8421 | 0.7516 | 6 | 32 | 1.9741 | 0.2852 |
| threshold_0.80_shrink_0.15 | 0.6500 | 38 | 0.8421 | 0.7508 | 6 | 32 | 1.9741 | 0.2862 |
| soft_cap_0.70_strength_0.25 | 0.6500 | 38 | 0.8421 | 0.7519 | 6 | 32 | 1.9513 | 0.2840 |
| soft_cap_0.70_strength_0.50 | 0.6500 | 38 | 0.8421 | 0.7332 | 6 | 32 | 1.8869 | 0.3087 |
| soft_cap_0.70_strength_0.75 | 0.6500 | 38 | 0.8421 | 0.7145 | 6 | 32 | 1.8297 | 0.3346 |
| soft_cap_0.75_strength_0.25 | 0.6500 | 38 | 0.8421 | 0.7603 | 6 | 32 | 1.9888 | 0.2732 |
| soft_cap_0.75_strength_0.50 | 0.6500 | 38 | 0.8421 | 0.7501 | 6 | 32 | 1.9560 | 0.2866 |
| soft_cap_0.75_strength_0.75 | 0.6500 | 38 | 0.8421 | 0.7398 | 6 | 32 | 1.9265 | 0.3004 |
| soft_cap_0.80_strength_0.25 | 0.6500 | 38 | 0.8421 | 0.7656 | 6 | 32 | 2.0112 | 0.2665 |
| soft_cap_0.80_strength_0.50 | 0.6500 | 38 | 0.8421 | 0.7607 | 6 | 32 | 1.9979 | 0.2729 |
| soft_cap_0.80_strength_0.75 | 0.6500 | 38 | 0.8421 | 0.7558 | 6 | 32 | 1.9856 | 0.2795 |

## Home Favourite Vs Away Favourite Behaviour

| favourite_type | n | accuracy | mean_confidence | mean_log_loss |
| --- | --- | --- | --- | --- |
| home_favourite | 63 | 0.6508 | 0.5965 | 0.8361 |
| draw_favourite | 4 | 0.5000 | 0.3598 | 1.0831 |
| away_favourite | 42 | 0.5952 | 0.5723 | 0.8924 |

## Worst Misses After Shrinkage

| match_date | round | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-12-07 | R10 | Chelsea | Everton | A | H | 0.8531 | 0.0450 | 3.1021 |
| 2025-11-16 | R9 | Liverpool | Chelsea | D | A | 0.7920 | 0.1470 | 1.9173 |
| 2025-12-14 | R11 | Leicester City | London City Lionesses | H | A | 0.5405 | 0.1489 | 1.9048 |
| 2026-03-15 | R17 | Aston Villa | Manchester City | D | A | 0.7550 | 0.1490 | 1.9038 |
| 2025-11-08 | R8 | Manchester United | Aston Villa | A | H | 0.6850 | 0.1530 | 1.8773 |
| 2026-05-04 | R21 | Aston Villa | West Ham United | A | H | 0.5540 | 0.1570 | 1.8515 |
| 2026-02-08 | R15 | West Ham United | Brighton | H | A | 0.5640 | 0.1580 | 1.8452 |
| 2026-04-26 | R20 | Liverpool | West Ham United | A | H | 0.4960 | 0.1590 | 1.8389 |
| 2025-11-02 | R7 | Aston Villa | Everton | D | H | 0.7550 | 0.1610 | 1.8264 |
| 2026-03-15 | R17 | Tottenham Hotspur | Everton | A | H | 0.5800 | 0.1780 | 1.7260 |

## Limitations

- This is one offline experiment over one generated comparison artefact.
- The sample contains 109 champion-evaluated matches, so differences should be treated as hypotheses.
- Fixed shrinkage variants are intentionally simple and do not learn a new production model.

## Recommendation

Recommendation: keep the original champion unchanged; no tested favourite shrinkage improved log loss.

- No favourite-shrinkage variant improved log loss or Brier score over the original champion.
- High-confidence miss count did not fall for the best variant.
- The original champion remains the best tested option, so no shrinkage variant should be promoted.
