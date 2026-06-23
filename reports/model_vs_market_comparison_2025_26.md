# WSL Model vs Market-Implied Benchmark

This is an evaluation-only matched-fixture comparison between existing model prediction artefacts and an external market probability reference. Market probabilities are derived from raw odds using proportional no-vig normalization and are not used as model features, training data or production prediction inputs.

## Summary

- Model fixtures: 109
- Market fixtures: 132
- Matched fixtures: 109
- Matched model rows: 436
- Unmatched model fixtures: 0
- Unmatched market fixtures: 23

## Metric Table

| rank | model_name | n_matches | brier_score | log_loss | accuracy | avg_actual_probability |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | market_implied_benchmark | 109 | 0.4885 | 0.8369 | 0.6330 | 0.5040 |
| 2 | champion_dc_xg | 109 | 0.5052 | 0.8669 | 0.6239 | 0.4805 |
| 3 | logistic_regression | 109 | 0.5562 | 0.9560 | 0.5780 | 0.4479 |
| 4 | elo_baseline | 109 | 0.5676 | 0.9597 | 0.5596 | 0.4059 |
| 5 | naive_outcome_rate | 109 | 0.6401 | 1.0576 | 0.4587 | 0.3629 |

## Disagreement Analysis

| model_name | disagreement_count | model_hit_rate_when_disagreeing | market_hit_rate_when_disagreeing | mean_max_probability_gap |
| --- | --- | --- | --- | --- |
| champion_dc_xg | 13 | 0.3077 | 0.3846 | 0.1385 |
| elo_baseline | 25 | 0.2000 | 0.5200 | 0.2049 |
| logistic_regression | 16 | 0.1875 | 0.5625 | 0.2234 |
| naive_outcome_rate | 45 | 0.1556 | 0.5778 | 0.2897 |

## Top Disagreements

| model_name | match_date | home_team | away_team | actual_outcome | model_pick | market_pick | model_actual_probability | market_actual_probability | log_loss_delta_model_minus_market | max_probability_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_outcome_rate | 2026-05-03 | Leicester City | Chelsea | A | H | A | 0.3583 | 0.9015 | 0.9226 | 0.5432 |
| elo_baseline | 2025-11-02 | Leicester City | Arsenal | A | H | A | 0.3482 | 0.8363 | 0.8762 | 0.4881 |
| naive_outcome_rate | 2025-12-07 | Leicester City | Manchester City | A | H | A | 0.3393 | 0.8138 | 0.8748 | 0.4745 |
| logistic_regression | 2025-10-12 | Chelsea | Tottenham Hotspur | H | A | H | 0.3364 | 0.8080 | 0.8762 | 0.4715 |
| naive_outcome_rate | 2025-11-02 | Leicester City | Arsenal | A | H | A | 0.3684 | 0.8363 | 0.8198 | 0.4679 |
| naive_outcome_rate | 2026-05-16 | West Ham United | Manchester City | A | H | A | 0.3594 | 0.8214 | 0.8266 | 0.4620 |
| naive_outcome_rate | 2026-04-26 | Everton | Chelsea | A | H | A | 0.3565 | 0.8126 | 0.8239 | 0.4561 |
| naive_outcome_rate | 2025-11-16 | Liverpool | Chelsea | D | H | A | 0.2200 | 0.1300 | -0.5259 | 0.4526 |
| logistic_regression | 2025-10-05 | West Ham United | Aston Villa | A | H | A | 0.2132 | 0.4346 | 0.7121 | 0.4410 |
| naive_outcome_rate | 2026-01-25 | London City Lionesses | Manchester City | A | H | A | 0.3200 | 0.7564 | 0.8603 | 0.4364 |

## Draw Sensitivity

| model_name | actual_draw_count | mean_model_draw_probability | mean_market_draw_probability | model_draw_pick_count | market_draw_pick_count | model_actual_draw_probability | market_actual_draw_probability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| champion_dc_xg | 21 | 0.2431 | 0.2014 | 4 | 0 | 0.2762 | 0.2233 |
| elo_baseline | 21 | 0.2140 | 0.2014 | 0 | 0 | 0.2263 | 0.2233 |
| logistic_regression | 21 | 0.1895 | 0.2014 | 0 | 0 | 0.1867 | 0.2233 |
| naive_outcome_rate | 21 | 0.1962 | 0.2014 | 0 | 0 | 0.1902 | 0.2233 |

## Market-Favourite Failure Analysis

| n_fixtures | market_favourite_wins | market_favourite_fails_to_win | market_underdog_wins |
| --- | --- | --- | --- |
| 109 | 69 | 40 | 19 |

## Model Less Confident In Failed Market Favourites

| model_name | match_date | home_team | away_team | actual_outcome | model_pick | market_pick | model_actual_probability | market_actual_probability | log_loss_delta_model_minus_market | max_probability_gap | market_favourite_probability_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_outcome_rate | 2025-11-16 | Liverpool | Chelsea | D | H | A | 0.2200 | 0.1300 | -0.5259 | 0.4526 | 0.4526 |
| naive_outcome_rate | 2025-12-07 | Chelsea | Everton | A | H | H | 0.3393 | 0.0532 | -1.8524 | 0.4118 | 0.4118 |
| naive_outcome_rate | 2026-03-15 | Aston Villa | Manchester City | D | H | A | 0.1753 | 0.1473 | -0.1741 | 0.4049 | 0.4049 |
| elo_baseline | 2025-11-16 | Tottenham Hotspur | Arsenal | D | H | A | 0.2275 | 0.1656 | -0.3177 | 0.4003 | 0.4003 |
| naive_outcome_rate | 2026-05-06 | Brighton | Arsenal | D | H | A | 0.1905 | 0.1545 | -0.2091 | 0.3815 | 0.3815 |
| elo_baseline | 2025-11-16 | Liverpool | Chelsea | D | A | A | 0.2319 | 0.1300 | -0.5786 | 0.3783 | 0.3783 |
| naive_outcome_rate | 2025-11-16 | Tottenham Hotspur | Arsenal | D | H | A | 0.2200 | 0.1656 | -0.2840 | 0.3677 | 0.3677 |
| naive_outcome_rate | 2026-03-18 | West Ham United | Manchester United | D | H | A | 0.1782 | 0.1820 | 0.0209 | 0.3505 | 0.3467 |
| naive_outcome_rate | 2026-03-21 | London City Lionesses | Chelsea | D | H | A | 0.1782 | 0.1813 | 0.0173 | 0.3487 | 0.3456 |
| naive_outcome_rate | 2026-04-25 | Brighton | Manchester City | H | H | A | 0.4522 | 0.1422 | -1.1565 | 0.3342 | 0.3342 |

## Model Outperformed Market By Row Log Loss

| model_name | match_date | home_team | away_team | actual_outcome | model_pick | market_pick | model_actual_probability | market_actual_probability | log_loss_delta_model_minus_market | max_probability_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_outcome_rate | 2025-12-07 | Chelsea | Everton | A | H | H | 0.3393 | 0.0532 | -1.8524 | 0.4118 |
| elo_baseline | 2025-12-07 | Chelsea | Everton | A | H | H | 0.2318 | 0.0532 | -1.4715 | 0.2447 |
| naive_outcome_rate | 2025-11-08 | Manchester United | Aston Villa | A | H | H | 0.3636 | 0.1027 | -1.2648 | 0.2732 |
| naive_outcome_rate | 2026-04-25 | Brighton | Manchester City | H | H | A | 0.4522 | 0.1422 | -1.1565 | 0.3342 |
| elo_baseline | 2025-11-08 | Manchester United | Aston Villa | A | H | H | 0.2745 | 0.1027 | -0.9837 | 0.1977 |
| logistic_regression | 2025-12-07 | Chelsea | Everton | A | H | H | 0.1238 | 0.0532 | -0.8441 | 0.1778 |
| elo_baseline | 2026-04-25 | Brighton | Manchester City | H | A | A | 0.3134 | 0.1422 | -0.7899 | 0.2226 |
| naive_outcome_rate | 2025-12-14 | Leicester City | London City Lionesses | H | H | A | 0.4355 | 0.2109 | -0.7250 | 0.2246 |
| naive_outcome_rate | 2026-02-08 | West Ham United | Brighton | H | H | A | 0.4419 | 0.2214 | -0.6911 | 0.2205 |
| elo_baseline | 2025-12-14 | Leicester City | London City Lionesses | H | H | A | 0.3951 | 0.2109 | -0.6277 | 0.1842 |

## Market Outperformed Model By Row Log Loss

| model_name | match_date | home_team | away_team | actual_outcome | model_pick | market_pick | model_actual_probability | market_actual_probability | log_loss_delta_model_minus_market | max_probability_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 2025-10-05 | Leicester City | Everton | D | H | A | 0.0612 | 0.2815 | 1.5252 | 0.2864 |
| logistic_regression | 2025-11-09 | West Ham United | Leicester City | D | H | H | 0.0786 | 0.2642 | 1.2127 | 0.1856 |
| logistic_regression | 2026-02-08 | Arsenal | Manchester City | H | A | H | 0.1436 | 0.3763 | 0.9635 | 0.2732 |
| naive_outcome_rate | 2026-05-03 | Leicester City | Chelsea | A | H | A | 0.3583 | 0.9015 | 0.9226 | 0.5432 |
| elo_baseline | 2025-11-02 | Leicester City | Arsenal | A | H | A | 0.3482 | 0.8363 | 0.8762 | 0.4881 |
| logistic_regression | 2025-10-12 | Chelsea | Tottenham Hotspur | H | A | H | 0.3364 | 0.8080 | 0.8762 | 0.4715 |
| naive_outcome_rate | 2025-12-07 | Leicester City | Manchester City | A | H | A | 0.3393 | 0.8138 | 0.8748 | 0.4745 |
| naive_outcome_rate | 2026-01-25 | London City Lionesses | Manchester City | A | H | A | 0.3200 | 0.7564 | 0.8603 | 0.4364 |
| naive_outcome_rate | 2026-05-16 | West Ham United | Manchester City | A | H | A | 0.3594 | 0.8214 | 0.8266 | 0.4620 |
| naive_outcome_rate | 2026-04-26 | Everton | Chelsea | A | H | A | 0.3565 | 0.8126 | 0.8239 | 0.4561 |

## Unmatched Fixtures

- Unmatched model fixtures: 0
- Unmatched market fixtures: 23

## Guardrails

- This is a matched-fixture comparison for evaluation only.
- Market odds are not used as model features or training data.
- This is not a market blending implementation.
- Interpret results only with source, timing and licensing limitations in mind.
