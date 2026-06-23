# Model vs Market-Implied Comparison

Phase 2 adds an offline, evaluation-only matched-fixture comparison between
existing model prediction artefacts and the market-implied benchmark from Phase
1. It does not use market odds as model features, does not use market odds for
training, does not blend market probabilities with model probabilities, and does
not change production prediction behaviour.

## Inputs

Primary model source:

```bash
reports/model_comparison_first_run.json
```

Primary market source:

```bash
data/exports/wsl_results_probabilities_2025_2026.csv
```

The market CSV remains a local ignored file. The comparison uses
`evaluation.market_benchmark.normalise_market_rows()` so market probabilities
continue to be derived from raw odds with proportional no-vig normalization.
Supplied `P_Home`, `P_Draw` and `P_Away` columns remain diagnostics only.

## Matching

The comparison uses matched fixtures only. The fixture key is:

```text
match_date_key, normalized_home_team, normalized_away_team
```

Dates are normalized to `YYYY-MM-DD`. Team names are stripped, case-folded and
then mapped through explicit aliases:

```python
{
    "manchester united": "manchester utd",
    "tottenham hotspur": "tottenham",
    "leicester city": "leicester",
    "west ham united": "west ham",
}
```

Scores and goals are not part of the merge key.

## Outputs

The runner produces:

- Combined metric table over matched fixtures only.
- Row-level model-vs-market comparison CSV.
- Disagreement analysis.
- Draw sensitivity analysis.
- Market-favourite failure and underdog outcome analysis.
- Worst row-level log-loss deltas in both directions.
- Markdown and JSON summaries.

## Runbook

```bash
python scripts/run_model_market_comparison.py \
  --model-json reports/model_comparison_first_run.json \
  --market-csv data/exports/wsl_results_probabilities_2025_2026.csv \
  --output-md reports/model_vs_market_comparison_2025_26.md \
  --output-json reports/model_vs_market_comparison_2025_26.json \
  --output-rows reports/model_vs_market_comparison_2025_26_rows.csv
```

## Interpretation Guardrails

Use careful language:

- market-implied benchmark
- external market probability reference
- evaluation-only comparison
- matched-fixture comparison

Avoid stronger claims unless odds source, snapshot timing and licensing are
verified. This Phase 2 report is an offline comparison layer, not a production
decision artifact and not a market blending implementation.
