# Data Contract

Last updated: 2026-06-19

This document defines the canonical data contract for future model evaluation and champion-vs-challenger experimentation. It documents current required fields separately from future optional fields so the existing prediction engine can remain stable while the evaluation framework matures.

## A. Required Champion Model Columns

These columns are required today by the current champion model in `model/wsl_xg_model.py` and by the Supabase RPC contract in `data/supabase_client.py`.

| Column | Canonical type | Required now | Description |
| --- | --- | --- | --- |
| `match_date` | date or datetime coercible to pandas datetime | Yes | Fixture date used for train/predict cutoffs and backtests. |
| `round_label` | string | Yes | Competition round or matchweek label, for example `R12`. |
| `home_team` | string | Yes | Home team display name. |
| `away_team` | string | Yes | Away team display name. |
| `home_xg` | float | Yes | Home team total xG. |
| `away_xg` | float | Yes | Away team total xG. |
| `home_np_xg` | float | Yes | Home team non-penalty xG. |
| `away_np_xg` | float | Yes | Away team non-penalty xG. |
| `home_goals` | integer or nullable integer | Yes | Home goals. Future fixtures may be null before completion. |
| `away_goals` | integer or nullable integer | Yes | Away goals. Future fixtures may be null before completion. |

The current model trains only on rows with valid historical xG and uses date-based cutoffs. Future rows may include actual result fields if the source is a complete historical table, but feature builders must ignore all result and xG values at or after the prediction cutoff.

## B. Evaluation Columns

Evaluation columns are required for replay evaluation, model comparison and reporting. Some are already available through `prediction_runs`; others should become standard as the framework expands.

| Column | Canonical type | Required now | Description |
| --- | --- | --- | --- |
| `actual_outcome` | string enum: `H`, `D`, `A` | Derived now | Final 1X2 result label. |
| `prediction_timestamp` | timezone-aware datetime | Future | When the prediction was generated. |
| `model_version` | string | Future | Code version, model registry ID or config version. |
| `prediction_run_id` | string or UUID | Current for logged runs | Identifier from `prediction_runs`. |
| `p_home_win` | float in `[0, 1]` | Current | Model probability for home win. |
| `p_draw` | float in `[0, 1]` | Current | Model probability for draw. |
| `p_away_win` | float in `[0, 1]` | Current | Model probability for away win. |
| `scoreline_probabilities` | JSON/object | Future optional | Full scoreline matrix or sparse scoreline probabilities when available. |

Evaluation outputs should record the exact fixture set, model config, data snapshot and code version used to generate the comparison.

## C. Market/Odds Columns

Market fields support external probability benchmarking and future model-plus-market blending. They are optional for pure model evaluation.

| Column | Canonical type | Required now | Description |
| --- | --- | --- | --- |
| `odds_source` | string | Future optional | Bookmaker, exchange or odds aggregator. |
| `odds_snapshot_timestamp` | timezone-aware datetime | Future optional | When odds were captured. |
| `home_odds` | float or fractional string | Market benchmark input | Home win odds. |
| `draw_odds` | float or fractional string | Market benchmark input | Draw odds. |
| `away_odds` | float or fractional string | Market benchmark input | Away win odds. |
| `odds_format` | string enum: `decimal`, `fractional` | Market benchmark input | Format of odds columns. |
| `raw_p_home` | float | Derived | Raw implied home probability before margin removal. |
| `raw_p_draw` | float | Derived | Raw implied draw probability before margin removal. |
| `raw_p_away` | float | Derived | Raw implied away probability before margin removal. |
| `fair_p_home` | float | Derived | Vig-removed fair home probability. |
| `fair_p_draw` | float | Derived | Vig-removed fair draw probability. |
| `fair_p_away` | float | Derived | Vig-removed fair away probability. |
| `odds_snapshot_label` | string | Future optional | Opening, closing, pre-match, in-week or manual snapshot label. |

Odds snapshot timing is analytically important. Closing prices, opening prices and manual midweek captures can produce materially different benchmark conclusions.

## D. Future Multi-League Columns

These fields are not required for the current WSL-only champion model, but they should become part of the canonical schema before PL-to-WSL transfer or multi-competition experiments.

| Column | Canonical type | Required now | Description |
| --- | --- | --- | --- |
| `season` | string, for example `2025-26` | Future recommended | Football season label. |
| `competition` | string | Future recommended | Competition name, for example `WSL`. |
| `league` | string | Future recommended | League label for multi-league experiments. |
| `match_id` | string or integer | Future recommended | Stable source match identifier. |
| `home_team_id` | string or integer | Future recommended | Stable home team identifier. |
| `away_team_id` | string or integer | Future recommended | Stable away team identifier. |
| `neutral_venue` | boolean | Future optional | Whether the fixture was played at a neutral venue. |

Stable IDs should be preferred over display names as soon as they are available.

## Matching-Key Strategy

Until stable `match_id` values are available, use this fallback matching key:

```text
season, round_label, match_date, normalized home_team, normalized away_team
```

If `season` is unavailable, use:

```text
round_label, match_date, normalized home_team, normalized away_team
```

Normalization should trim whitespace and compare team names case-insensitively. This is a fallback only. String-based matching can fail when team names change, aliases differ, or source systems use different naming conventions.

## Known Risks

- String-based team matching is fragile.
- Rescheduled fixtures can create date/round ambiguity.
- Round labels are useful for replay validation but should not replace date-based as-of logic.
- A complete historical source table may contain future xG/goals relative to a prediction date; feature builders must not consume those values.
- Market odds snapshots must be timestamped or labelled clearly to avoid comparing model predictions against odds captured with different information.

## Leakage Prevention Principles

Future feature builders must follow these rules:

1. Accept an explicit `as_of_date`, `as_of_round`, or fold cutoff.
2. Use only matches strictly before the cutoff for training features.
3. Never aggregate goals, xG, form, standings or market movement from the prediction fixture or future fixtures.
4. Persist fold definitions so every model is evaluated on the same train/test rows.
5. Store model version, data snapshot and feature configuration with each evaluation artifact.
6. Treat Week 1 predictions as a special case requiring previous-season priors or a historical baseline.
