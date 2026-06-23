# WSL Market-Implied Benchmark

This report derives proportional no-vig probabilities directly from the raw fractional odds columns (`Odds_1`, `Odds_X`, `Odds_2`) and evaluates them as an external market probability reference. The supplied de-vigged probability columns are used only for data-quality diagnostics. This layer is evaluation-only and is not used for model training or features.

## Summary

- Benchmark: `market_implied_benchmark`
- Fixture count: 132
- Brier score: 0.471181
- Log loss: 0.812748
- Accuracy: 0.651515
- Raw rows: 133
- Evaluated rows: 132
- Excluded non-league rows: 1

## Data Quality Warnings

| type | count | threshold | max_abs_provided_vs_derived_diff | message |
| --- | --- | --- | --- | --- |
| underround_rows | 3 |  |  |  |
| provided_vs_derived_probability_mismatch | 1 | 0.0500 |  |  |
| provided_probability_reference_check |  |  | 0.2400 | Benchmark probabilities are derived from raw odds. Supplied P_Home/P_Draw/P_Away values are retained only as diagnostics. |
| non_league_rows_excluded | 1 |  |  |  |

## Metric Table

| rank | model_name | n_matches | brier_score | log_loss | accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | market_implied_benchmark | 132 | 0.4712 | 0.8127 | 0.6515 |

## Calibration

| bin | count | mean_confidence | observed_accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| 0%-20% | 0 |  |  |  | True |
| 20%-40% | 15 | 0.3793 | 0.4667 | -0.0874 | False |
| 40%-60% | 49 | 0.4882 | 0.4694 | 0.0188 | False |
| 60%-80% | 43 | 0.7182 | 0.7907 | -0.0725 | False |
| 80%-100% | 25 | 0.8407 | 0.8800 | -0.0393 | False |

## Confidence Buckets

| bucket | count | mean_confidence | accuracy | calibration_gap | is_sparse |
| --- | --- | --- | --- | --- | --- |
| low | 15 | 0.3793 | 0.4667 | -0.0874 | False |
| medium | 49 | 0.4882 | 0.4694 | 0.0188 | False |
| high | 68 | 0.7632 | 0.8235 | -0.0603 | False |

## Worst Misses

| match_date | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-12-07 | Chelsea | Everton | A | H | 0.8403 | 0.0532 | 2.9333 |
| 2025-11-08 | Manchester Utd | Aston Villa | A | H | 0.7278 | 0.1027 | 2.2764 |
| 2025-09-27 | Arsenal | Aston Villa | D | H | 0.8325 | 0.1117 | 2.1923 |
| 2025-11-16 | Liverpool | Chelsea | D | A | 0.8126 | 0.1300 | 2.0401 |
| 2026-04-25 | Brighton | Manchester City | H | A | 0.6908 | 0.1422 | 1.9502 |
| 2026-03-15 | Aston Villa | Manchester City | D | A | 0.7657 | 0.1473 | 1.9156 |
| 2026-05-06 | Brighton | Arsenal | D | A | 0.7386 | 0.1545 | 1.8673 |
| 2025-11-16 | Tottenham | Arsenal | D | A | 0.7277 | 0.1656 | 1.7981 |
| 2026-03-21 | London City Lionesses | Chelsea | D | A | 0.7119 | 0.1813 | 1.7074 |
| 2026-03-18 | West Ham | Manchester Utd | D | A | 0.7130 | 0.1820 | 1.7038 |

## Notes

- Do not interpret this as evidence that any model beats bookmakers.
- Final published conclusions require verified odds source, snapshot timing and licensing.
