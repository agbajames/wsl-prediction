# Champion Draw-Adjustment Experiment

## Executive Summary

This offline experiment tests fixed draw adjustments for `champion_dc_xg` on 109 evaluated matches with 21 actual draws.
Best variant by log loss: `additive_draw_-0.050`. Deltas versus original: Brier -0.0041, log loss -0.0088, accuracy -0.0092.
Recommendation: consider this draw-adjusted variant for future offline production-style testing, while keeping the original champion as the reference.

## Methodology

- Apply fixed additive and multiplicative draw adjustments to champion probabilities.
- Additive deltas tested: -0.05, -0.025, 0.025, 0.05
- Multiplicative factors tested: 0.85, 0.95, 1.05, 1.15
- The underlying champion model, API, dashboard, and production outputs are unchanged.

## Original Champion Draw Behaviour

| actual_draws | draw_prediction_rate | avg_predicted_draw_probability | draw_recall | draw_log_loss | non_draw_log_loss | non_draw_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | 0.0367 | 0.2431 | 0.0952 | 1.3243 | 0.7578 | 0.7500 |

## Draw-Adjustment Variants Tested

| variant_name | method | value | brier_score | log_loss | accuracy | draw_prediction_rate | avg_predicted_draw_probability | draw_recall | draw_log_loss | non_draw_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_champion | original | 0.0000 | 0.5052 | 0.8669 | 0.6239 | 0.0367 | 0.2431 | 0.0952 | 1.3243 | 0.7578 |
| additive_draw_-0.050 | additive | -0.0500 | 0.5011 | 0.8581 | 0.6147 | 0.0000 | 0.1931 | 0.0000 | 1.5464 | 0.6939 |
| additive_draw_-0.025 | additive | -0.0250 | 0.5021 | 0.8607 | 0.6147 | 0.0000 | 0.2181 | 0.0000 | 1.4283 | 0.7253 |
| additive_draw_+0.025 | additive | 0.0250 | 0.5104 | 0.8760 | 0.6239 | 0.0642 | 0.2681 | 0.1429 | 1.2310 | 0.7913 |
| additive_draw_+0.050 | additive | 0.0500 | 0.5177 | 0.8877 | 0.5963 | 0.1284 | 0.2931 | 0.1905 | 1.1462 | 0.8261 |
| multiplicative_draw_0.85 | multiplicative | 0.8500 | 0.5021 | 0.8610 | 0.6147 | 0.0000 | 0.2153 | 0.0000 | 1.4444 | 0.7218 |
| multiplicative_draw_0.95 | multiplicative | 0.9500 | 0.5039 | 0.8645 | 0.6330 | 0.0183 | 0.2341 | 0.0952 | 1.3617 | 0.7459 |
| multiplicative_draw_1.05 | multiplicative | 1.0500 | 0.5067 | 0.8696 | 0.6147 | 0.0459 | 0.2519 | 0.0952 | 1.2892 | 0.7694 |
| multiplicative_draw_1.15 | multiplicative | 1.1500 | 0.5104 | 0.8757 | 0.6239 | 0.0642 | 0.2688 | 0.1429 | 1.2251 | 0.7924 |

## Original Champion Vs Best Draw-Adjusted Variant

| variant_name | brier_score | log_loss | accuracy | draw_recall | draw_log_loss |
| --- | --- | --- | --- | --- | --- |
| original_champion | 0.5052 | 0.8669 | 0.6239 | 0.0952 | 1.3243 |
| additive_draw_-0.050 | 0.5011 | 0.8581 | 0.6147 | 0.0000 | 1.5464 |

## Overall Metrics

| variant_name | brier_score | log_loss | accuracy | draw_recall | draw_log_loss |
| --- | --- | --- | --- | --- | --- |
| additive_draw_-0.050 | -0.0041 | -0.0088 | -0.0092 | -0.0952 | 0.2221 |

## Draw-Specific Metrics

| variant_name | actual_draws | draw_prediction_rate | avg_predicted_draw_probability | draw_recall | draw_log_loss | non_draw_log_loss | non_draw_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_champion | 21 | 0.0367 | 0.2431 | 0.0952 | 1.3243 | 0.7578 | 0.7500 |
| additive_draw_-0.050 | 21 | 0.0000 | 0.1931 | 0.0000 | 1.5464 | 0.6939 | 0.7614 |
| additive_draw_-0.025 | 21 | 0.0000 | 0.2181 | 0.0000 | 1.4283 | 0.7253 | 0.7614 |
| additive_draw_+0.025 | 21 | 0.0642 | 0.2681 | 0.1429 | 1.2310 | 0.7913 | 0.7386 |
| additive_draw_+0.050 | 21 | 0.1284 | 0.2931 | 0.1905 | 1.1462 | 0.8261 | 0.6932 |
| multiplicative_draw_0.85 | 21 | 0.0000 | 0.2153 | 0.0000 | 1.4444 | 0.7218 | 0.7614 |
| multiplicative_draw_0.95 | 21 | 0.0183 | 0.2341 | 0.0952 | 1.3617 | 0.7459 | 0.7614 |
| multiplicative_draw_1.05 | 21 | 0.0459 | 0.2519 | 0.0952 | 1.2892 | 0.7694 | 0.7386 |
| multiplicative_draw_1.15 | 21 | 0.0642 | 0.2688 | 0.1429 | 1.2251 | 0.7924 | 0.7386 |

## Worst Misses After Draw Adjustment

| match_date | round | home_team | away_team | actual_outcome | predicted_outcome | predicted_confidence | actual_probability | row_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-12-07 | R10 | Chelsea | Everton | A | H | 0.9006 | 0.0475 | 3.0479 |
| 2025-11-16 | R9 | Liverpool | Chelsea | D | A | 0.8384 | 0.0970 | 2.3330 |
| 2026-03-15 | R17 | Aston Villa | Manchester City | D | A | 0.7994 | 0.0990 | 2.3126 |
| 2025-11-02 | R7 | Aston Villa | Everton | D | H | 0.8000 | 0.1110 | 2.1982 |
| 2025-10-03 | R5 | Manchester United | Chelsea | D | H | 0.5946 | 0.1298 | 2.0416 |
| 2025-12-14 | R11 | Leicester City | London City Lionesses | H | A | 0.5797 | 0.1596 | 1.8348 |
| 2025-11-08 | R8 | Manchester United | Aston Villa | A | H | 0.7259 | 0.1621 | 1.8194 |
| 2025-11-16 | R9 | Tottenham Hotspur | Arsenal | D | A | 0.7061 | 0.1672 | 1.7885 |
| 2026-05-04 | R21 | Aston Villa | West Ham United | A | H | 0.5930 | 0.1680 | 1.7835 |
| 2026-02-08 | R15 | West Ham United | Brighton | H | A | 0.6031 | 0.1689 | 1.7782 |

## Limitations

- This is one offline experiment over one generated comparison artefact.
- The sample contains 109 champion-evaluated matches, so differences should be treated as hypotheses.
- Fixed adjustments are intentionally simple and do not learn a new production model.

## Recommendation

Recommendation: consider this draw-adjusted variant for future offline production-style testing, while keeping the original champion as the reference.

- The best draw-adjusted variant improved at least one probability-quality metric.
- The winning variant shrinks draw probability, suggesting the champion may over-price draws on non-draws in this artefact.
- Actual-draw handling got worse, so the overall gain comes from non-draw fixtures rather than better draw recognition.
- Accuracy fell, but this can still be useful when Brier or log loss improves because probability quality matters.
