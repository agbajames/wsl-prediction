# Market Benchmark Strategy

Phase 1 adds an evaluation-only market-implied benchmark layer for WSL 2025-26
1X2 odds data. The benchmark derives proportional no-vig probabilities directly
from raw odds and treats them as an external market probability reference. It
does not train a model, generate model features, change production prediction
behaviour or make production betting claims.

## Scope

Included in Phase 1:

- Load the local market odds CSV.
- Validate required columns, result fields and probability discipline.
- Parse fractional or decimal 1X2 odds.
- Derive raw implied probabilities and overround from `Odds_1`, `Odds_X` and
  `Odds_2`.
- Score odds-derived proportional no-vig probabilities with the existing Brier
  score, log loss, accuracy and calibration helpers.
- Retain supplied `P_Home`, `P_Draw` and `P_Away` values only as diagnostic
  reference columns.
- Generate Markdown, JSON and row-level CSV report artefacts.
- Keep the full market odds file out of Git.

Excluded from Phase 1:

- Model-vs-market comparison.
- Market/model blending.
- Market odds as model features.
- Market odds as training inputs.
- Production API or dashboard behaviour changes.

## Data Placement

The full 2025-26 benchmark file should remain local:

```bash
data/exports/wsl_results_probabilities_2025_2026.csv
```

`data/exports/*.csv` remains ignored by Git. The committed test fixture is the
small, labelled sample at:

```bash
data/samples/market_odds_benchmark_sample.csv
```

## Input Contract

The provided CSV shape uses these required columns:

| column | meaning |
| --- | --- |
| `Date` | Match date. |
| `Home_Team`, `Away_Team` | Team display names. |
| `Home_Goals`, `Away_Goals` | Completed result used to derive H/D/A outcome. |
| `Odds_1`, `Odds_X`, `Odds_2` | Home/draw/away raw 1X2 odds, fractional or decimal. |
| `Imp_Home`, `Imp_Draw`, `Imp_Away` | Supplied raw implied probabilities, used for diagnostics. |
| `Overround` | Supplied raw implied probability sum, used for diagnostics. |
| `P_Home`, `P_Draw`, `P_Away` | Supplied de-vigged probabilities, used for diagnostics. |
| `Note` | Optional row note. Non-empty notes are excluded by default. |

Rows with a non-empty `Note`, such as a relegation play-off, are excluded from
the league-only benchmark unless `--include-non-league` is passed.

The benchmark probabilities are derived as:

```text
decimal odds = fractional odds + 1
imp_home = 1 / decimal_home
imp_draw = 1 / decimal_draw
imp_away = 1 / decimal_away
overround = imp_home + imp_draw + imp_away
p_home_win = imp_home / overround
p_draw = imp_draw / overround
p_away_win = imp_away / overround
```

The supplied de-vigged columns are retained as `provided_p_home`,
`provided_p_draw` and `provided_p_away`. The odds-derived diagnostic columns are
`derived_p_home`, `derived_p_draw`, `derived_p_away` and
`max_abs_provided_vs_derived_diff`. Rows with a supplied-vs-derived probability
difference greater than `0.05` are flagged.

## Runbook

Generate the league-only benchmark report from the ignored full CSV:

```bash
python scripts/run_market_benchmark.py \
  --csv data/exports/wsl_results_probabilities_2025_2026.csv \
  --output-md reports/market_benchmark_2025_26.md \
  --output-json reports/market_benchmark_2025_26.json \
  --output-rows reports/market_benchmark_2025_26_rows.csv
```

The generated reports are local artefacts until their source, timing and
licensing are reviewed.

## Interpretation Guardrails

Use precise language:

- market-implied benchmark
- external market probability reference
- de-vigged market probability benchmark
- evaluation-only benchmark layer

Avoid overclaiming:

- Keep model-vs-market conclusions limited to the matched-fixture data and
  documented odds provenance.
- Do not compare model predictions against market probabilities until fixture
  matching and snapshot timing are verified.
- Do not publish final conclusions if severe supplied-vs-odds-derived
  probability quality warnings remain unresolved.

Odds snapshot timing is analytically important. Opening, closing and manual
midweek captures can produce different benchmark conclusions.

## Phase 2 Handoff

The Phase 2 model-vs-market comparison should call the Phase 1 loader and
normalizer rather than reading supplied `P_Home`, `P_Draw` and `P_Away` as
benchmark probabilities. It should use matched fixtures only, with normalized
date/home/away keys and explicit team aliases documented in
`docs/model_vs_market_comparison.md`.
