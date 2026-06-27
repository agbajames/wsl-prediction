# Champion Calibration and Decay Plan

Date: 2026-06-26

## Executive Summary

The next useful model-improvement work should stay close to the champion family. The best current probability-quality signal is `dc_fit_rho_each_fold`, with log loss 0.8623 and Brier 0.5027 on the shared 109-fixture local comparison window. The unchanged operational reference remains `champion_dc_xg`, with log loss 0.8669 and Brier 0.5052. The most balanced champion-family candidate is `txg_xg_pseudocount_010`, with log loss 0.8649, Brier 0.5038, and accuracy 0.6330.

The gains are real enough to justify a careful follow-up, but too small to justify promotion. This branch predeclares a narrow experiment set around fitted Dixon-Coles rho, xG pseudocount/shrinkage, conservative time decay, calibration diagnostics, and fixed champion-family blends.

Recommended first implementation deliverable after this plan: a consolidated champion-family experiment runner/report that reuses existing variant wrappers and adds diagnostics for fitted rho by fold, pseudocount sensitivity, decay sensitivity, draw calibration, overconfidence, and champion-family blends.

## Evidence Baseline

All rows below refer to the common 2025-10-01 to 2026-05-16 local comparison window with 109 evaluated fixtures unless labelled otherwise.

| model_or_variant | type | log_loss | brier_score | accuracy | status |
| --- | --- | --- | --- | --- | --- |
| `dc_fit_rho_each_fold` | Champion-family candidate; fitted rho inside each training fold | 0.8623 | 0.5027 | 0.6147 | Best current probability-quality candidate; shadow-ready hypothesis, not production-ready |
| `txg_xg_pseudocount_010` | Champion-family candidate; xG pseudocount 0.10 | 0.8649 | 0.5038 | 0.6330 | Most balanced candidate; shadow-ready hypothesis, not production-ready |
| `blend_dc_fit_txg_50_50` | Fixed champion-family blend | 0.8634 | 0.5031 | 0.6239 | Best blend; useful carry-forward candidate, not standalone proof |
| `champion_dc_xg` | Operational/reference champion | 0.8669 | 0.5052 | 0.6239 | Frozen operational reference |
| `dc_rho_mild_minus_08` | Fixed-rho Dixon-Coles variant | 0.8650 | 0.5041 | 0.6330 | Exploratory support for rho sensitivity |
| `dc_rho_stronger_minus_18` | Fixed-rho Dixon-Coles variant | 0.8691 | 0.5064 | 0.6147 | Exploratory; worse than champion |
| `dc_conservative_xg_shrinkage` | Larger xG pseudocount plus stronger penalty shrinkage | 0.8669 | 0.5052 | 0.6239 | Exploratory; effectively tied with champion |
| `dc_score_grid_10` | Larger scoreline grid | 0.8670 | 0.5052 | 0.6239 | Park unless a later diagnostic shows truncation errors |
| `dc_alpha_030` | Stronger ridge shrinkage | 0.8681 | 0.5058 | 0.6147 | Exploratory; worse than champion |
| `txg_decay_90d` | Longer decay half-life | 0.8672 | 0.5055 | 0.6147 | Conservative decay diagnostic; not currently leading |
| `txg_alpha_025` | Moderately stronger ridge shrinkage | 0.8677 | 0.5056 | 0.6239 | Exploratory; not currently leading |
| `txg_conservative_weighting` | Combined alpha/pseudocount/penalty shrinkage | 0.8682 | 0.5058 | 0.6239 | Exploratory; not currently leading |
| `txg_decay_45d` | Shorter decay half-life | 0.8684 | 0.5061 | 0.6239 | Park aggressive recency unless new evidence supports it |
| `additive_draw_-0.050` | Post-processing draw shrinkage | 0.8581 | 0.5011 | 0.6147 | Interesting but risky; improves overall metrics by hurting actual-draw handling |
| `calibrated_champion` | Temperature plus base-rate shrinkage | 0.7951 on 43-match trial split | 0.4531 on 43-match trial split | 0.6512 on 43-match trial split | Park this broad calibration form; it worsened the original on its trial split |

Calibration and draw rows are not directly comparable to the main full-window candidate leaderboard unless labelled carefully. The calibration run used an earlier/later split: original champion on the 43-match trial split had Brier 0.4503, log loss 0.7831, and accuracy 0.6744, while calibrated champion worsened to Brier 0.4531, log loss 0.7951, and accuracy 0.6512.

The draw-adjustment result is also a caution. `additive_draw_-0.050` improved full-window Brier and log loss but reduced accuracy and drove draw recall to 0.0000, with draw log loss worsening from 1.3243 to 1.5464. It is a diagnostic signal about draw overpricing on non-draws, not a promotion candidate by itself.

Market comparison is separate. On 109 matched fixtures, `market_implied_benchmark` scored log loss 0.8369 and Brier 0.4885 versus `champion_dc_xg` at 0.8669 and 0.5052. Market odds remain an external benchmark only, not features for this branch.

## Current Implementation Map

| concern | current implementation |
| --- | --- |
| Frozen operational champion | `models/champion_dc_xg.py` wraps `model/wsl_xg_model.py` without copying production logic |
| Champion defaults | `ModelConfig`: `alpha=0.15`, `decay_half_life_days=60.0`, `rho=-0.13`, `max_goals=8`, `home_advantage_prior=0.25`, `penalty_shrinkage_n=5.0`, `xg_pseudocount=0.05`, `rho_grid=(-0.30, 0.01, 0.01)` |
| Team strengths | `estimate_team_strengths()` uses weighted ridge regression on log non-penalty xG, time decay, xG pseudocount, and a lightly regularised home advantage coefficient |
| Penalty rates | `estimate_penalty_rates()` uses home/away penalty xG with James-Stein shrinkage toward league averages |
| Rho fitting | `fit_rho()` searches the configured rho grid using actual scoreline likelihood after xG strengths are fixed |
| Variant wrapper | `evaluation/dixon_coles_variants.py` defines `DixonColesVariantSpec` and `DixonColesVariantModel` over `ChampionDCXGModel` |
| Time-decay/xG variants | `evaluation/time_decay_xg_variants.py` defines the existing small recency/xG weighting grid |
| Fixed blends | `evaluation/blending.py` defines fixed non-market blends including `blend_dc_fit_txg_50_50` |
| Shared folds | `evaluation/backtesting.py` builds deterministic date-based rolling folds with train rows strictly before test windows |
| Shared metrics | `evaluation/compare.py` ranks by log loss, then Brier, then accuracy |

The safest extension point is the variant wrapper layer, not the champion implementation. Candidate models should remain separate named variants/adapters/configs.

## What Has Already Been Tried

### Dixon-Coles Rho

Already tested:

- Frozen champion fixed rho `-0.13`.
- Fixed rho `-0.08`.
- Fixed rho `-0.18`.
- Fitted rho inside each fold through `dc_fit_rho_each_fold`.

Result: fitted rho is the best probability-quality candidate so far. Fixed `-0.08` also improved over champion but not as much as fitted rho by log loss/Brier.

Missing diagnostic: rho values by fold, train size, and fold-level delta versus champion. The current reports show aggregate metrics, not whether fitted rho is stable or just exploiting a few folds.

### xG Pseudocount and Shrinkage

Already tested:

- Champion default xG pseudocount `0.05`.
- `txg_xg_pseudocount_010` with pseudocount `0.10`.
- Combined conservative weighting with alpha `0.25`, xG pseudocount `0.10`, and penalty shrinkage `10`.
- Stronger alpha-only variants.

Result: `xg_pseudocount=0.10` is a useful signal. Combining multiple shrinkage changes was worse than the simple pseudocount variant, so the next branch should isolate effects rather than stack changes.

### Time Decay

Already tested:

- Champion 60-day decay.
- 45-day decay.
- 90-day decay.
- Earlier Dixon-Coles 30-day decay.

Result: neither shorter nor longer decay beat the leading fitted-rho/pseudocount variants. The conservative 90-day row was closer than 45-day, but neither clearly improved probability quality. Aggressive decay should be parked unless fold-level diagnostics show a specific failure mode.

### Draw and Probability Calibration

Already tested:

- Temperature scaling plus base-rate shrinkage.
- Fixed additive and multiplicative draw adjustments.

Result: broad calibration worsened Brier/log loss/accuracy on its trial split. Draw shrinkage improved overall log loss and Brier on the full champion artifact but worsened actual-draw handling and reduced accuracy. Draw work is still diagnostically relevant, but any future post-processing must report draw recall, draw log loss, non-draw log loss, top-class accuracy, and calibration bands.

### Fixed Champion-Family Blends

Already tested:

- Several champion-led blends with regularised team strength, improved logistic, and random forest.
- `blend_dc_fit_txg_50_50` between fitted rho and xG pseudocount.

Result: `blend_dc_fit_txg_50_50` is the only blend worth carrying forward in this branch. It beat the champion but did not beat `dc_fit_rho_each_fold`.

## What Should Continue

Continue with a narrow, predeclared champion-family experiment:

- Fitted rho diagnostics.
- Pseudocount sensitivity around `0.10`.
- Conservative decay checks as diagnostics, not a large search.
- Isolated shrinkage and penalty-rate diagnostics.
- Draw calibration diagnostics with draw-specific metrics.
- Fixed blends among only `champion_dc_xg`, `dc_fit_rho_each_fold`, and `txg_xg_pseudocount_010`.

## What Should Be Parked

Park for this branch:

- Neural networks, PyTorch, embeddings, deeper NN architectures, or larger NN grids.
- Broad calibration optimizers fit on the same 109-match window.
- Aggressive recency decay unless a diagnostic clearly justifies it.
- Scoreline grid expansion unless truncation error is shown.
- Market odds as features.
- Broad multi-dimensional grids combining rho, decay, alpha, pseudocount, penalty shrinkage, draw adjustment, and blend weights.
- Promotion decisions from one offline window.

## Predeclared Candidate Experiment Set

The next implementation should build a single consolidated runner/report, for example `scripts/run_champion_family_experiments.py`, that uses the existing rolling folds and wrapper style.

### A. Fitted Dixon-Coles Rho

Candidates:

| name | change |
| --- | --- |
| `champion_dc_xg` | Fixed reference rho `-0.13` |
| `dc_rho_mild_minus_08` | Fixed rho `-0.08`, retained as a simple fixed-rho comparator |
| `dc_fit_rho_each_fold` | Fit rho inside each training fold |

Required diagnostics:

- Fitted rho value per fold.
- Fold train size and test size.
- Fold log loss/Brier/accuracy for fitted rho and champion.
- Fold-level delta: fitted rho minus champion for log loss and Brier.
- Rho distribution: min, max, mean, median, standard deviation.
- Count of folds where fitted rho hits the grid boundary.
- Whether fitted rho improves broadly or only on one or two folds.

Go-forward criterion: fitted rho remains interesting only if it improves log loss or Brier across multiple folds and does not rely on boundary fits.

### B. xG Pseudocount and Shrinkage

Candidates:

| name | xg_pseudocount | notes |
| --- | --- | --- |
| `champion_dc_xg` | 0.05 | Reference |
| `txg_xg_pseudocount_005` | 0.05 | Same as reference if used only in pseudocount grid; can be omitted if champion row is present |
| `txg_xg_pseudocount_010` | 0.10 | Current best balanced candidate |
| `txg_xg_pseudocount_015` | 0.15 | Small follow-up above current best |
| `txg_xg_pseudocount_020` | 0.20 | Upper conservative sensitivity bound |

Optional only if needed for clarity:

- `txg_xg_pseudocount_0075`
- `txg_xg_pseudocount_0125`

We will only Use the optional half-steps if the four-point grid suggests a smooth local optimum and the branch explicitly needs a sensitivity plot.

Required diagnostics:

- Aggregate log loss/Brier/accuracy.
- Fold-level deltas versus champion and versus `dc_fit_rho_each_fold`.
- Mean predicted max confidence.
- High-confidence wrong count.
- Actual-outcome probability below 5% and 10%.
- Team-level or favourite-type regressions if pseudocount materially changes confidence.

Go-forward criterion: pseudocount `0.10` or a nearby value must improve probability quality without creating worse overconfidence or degrading draw handling badly.

### C. Time Decay and Recency Weighting

Candidates:

| name | decay_half_life_days | notes |
| --- | --- | --- |
| `champion_dc_xg` | 60 | Reference |
| `txg_decay_75d` | 75 | Conservative midpoint between current and 90-day |
| `txg_decay_90d` | 90 | Existing conservative longer-decay variant |
| `txg_decay_45d` | 45 | Existing shorter-decay diagnostic; include only to verify it remains worse |

Do not include 30-day decay in the main candidate set unless specifically diagnosing recency overreaction. It already underperformed as an aggressive short-decay idea.

Required diagnostics:

- Aggregate probability metrics.
- Fold-level deltas.
- Whether gains are concentrated in late-season or rescheduled fixture windows.
- Accuracy versus log-loss tradeoff.

Go-forward criterion: continue only if a conservative decay setting improves log loss or Brier without obvious fold concentration.

### D. Attack/Defence Strength Caps and Shrinkage

Do not implement caps immediately. First add diagnostics:

- Per-fold attack/defence parameter ranges.
- Teams with extreme attack or defence values.
- Expected-goals lambda ranges by fixture.
- Relationship between high-confidence misses and extreme team strengths.

Only if diagnostics show parameter extremes driving overconfidence should we add conservative candidate variants, such as:

- soft clipping attack/defence effects after fitting;
- stronger alpha near `0.20` or `0.25`;
- cap only as an offline candidate wrapper, not in the champion implementation.

### E. Penalty xG and Home/Away Penalty Rates

Current handling:

- Penalty xG is `xG - np_xG`, clipped at zero.
- Home and away penalty rates are estimated separately.
- Rates are shrunk toward league averages using `penalty_shrinkage_n`, default `5.0`.

Plan:

- Add diagnostics before tuning: league home/away penalty xG means, per-team home/away penalty sample sizes, and shrunk rates.
- Compare whether high-confidence misses are sensitive to penalty components.
- Keep `penalty_shrinkage_n=10.0` only as part of an isolated penalty-shrinkage diagnostic if sample sizes look unstable.

Do not overfit penalty rates on this one-season sample.

### F. Home Advantage

Current handling:

- Home advantage is estimated as a coefficient inside the weighted ridge regression.
- The home advantage coefficient is lightly regularised through the same `alpha` term.
- `home_advantage_prior` exists in `ModelConfig`, but the current strength fitting code regularises the home advantage coefficient directly with `alpha * 0.5`.

Plan:

- Add diagnostics for fitted home advantage per fold.
- Report exponentiated home advantage multiplier per fold.
- Compare fold-level home favourite and away favourite performance.
- Do not tune home advantage until the diagnostic shows instability or systematic home/away bias.

### G. Draw Calibration and Probability Calibration

Candidate set:

| name | type | status |
| --- | --- | --- |
| `original_champion` | reference | Keep as calibration baseline |
| `additive_draw_-0.025` | mild draw shrinkage | Plausible diagnostic; less damaging than -0.050 |
| `multiplicative_draw_0.95` | mild draw shrinkage | Plausible diagnostic; preserved some accuracy in first run |
| `additive_draw_-0.050` | stronger draw shrinkage | Include only as existing best-log-loss reference, not as preferred candidate |

Park:

- temperature plus base-rate shrinkage in its current form;
- draw-increasing variants unless the branch explicitly optimizes actual-draw handling.

Required diagnostics:

- Overall log loss/Brier/accuracy.
- Actual draw count.
- Draw prediction rate.
- Average predicted draw probability.
- Draw recall.
- Draw log loss.
- Non-draw log loss.
- Non-draw accuracy.
- Calibration bands and confidence buckets.

Go-forward criterion: draw post-processing must not be considered healthy if it improves overall log loss only by assigning nearly no probability to true draws.

### H. Fixed Champion-Family Blends

Candidates:

| name | components | weights |
| --- | --- | --- |
| `blend_dc_fit_txg_50_50` | `dc_fit_rho_each_fold`, `txg_xg_pseudocount_010` | 0.50 / 0.50 |
| `blend_champion_dc_fit_50_50` | `champion_dc_xg`, `dc_fit_rho_each_fold` | 0.50 / 0.50 |
| `blend_champion_txg_50_50` | `champion_dc_xg`, `txg_xg_pseudocount_010` | 0.50 / 0.50 |
| `blend_champion_dc_fit_txg_34_33_33` | all three champion-family references | approximately equal weights |

Do not include:

- neural networks;
- market odds;
- broad learned blend weights;
- feature-ML blends in this branch unless the branch scope is explicitly expanded.

## Metrics and Ranking Rules

Primary ranking:

1. Log loss.
2. Brier score.
3. Accuracy.

Required supporting diagnostics:

- Fold-level metrics and deltas.
- Calibration bands.
- Confidence buckets.
- High-confidence wrong predictions.
- Actual-outcome probability below 5% and 10%.
- Draw-specific metrics.
- Favourite breakdown: home favourite, away favourite, predicted draw.
- Team-level error summaries for recurring problem teams.

Accuracy is useful context, but it must not override log loss and Brier score.

## Validation and Fold Protocol

Use the existing shared rolling backtest protocol unless a report clearly labels another setup:

- CSV: local WSL match export with real xG columns.
- Test start: 2025-10-01.
- Test end: 2026-05-16.
- Minimum training matches: 12, matching the latest corrected experiment convention where possible.
- Test window: 7 days.
- Step: 7 days.
- Train rows must have `match_date < test_start`.
- Use identical folds for all candidates.
- Include fold metadata in JSON outputs.

If an experiment uses a different trial split, such as earlier-fold calibration and later-fold validation, label it separately and do not mix it into the main leaderboard.

## Leakage Guardrails

- Keep `model/wsl_xg_model.py` and `models/champion_dc_xg.py` frozen.
- Implement candidates as separate variants/adapters/configs.
- Use only historical rows before each fold test window.
- Do not use market odds as model features.
- Do not use full-season aggregates or future-derived features.
- Do not fabricate shadow fixtures.
- Keep grids small and predeclared.
- Treat tiny metric differences as hypotheses, not proof.
- Keep raw/private exports and `.env` out of commits.

## Candidate Promotion Criteria

A candidate can become shadow-ready if it:

- improves log loss or Brier versus `champion_dc_xg` on the shared folds;
- does not materially worsen calibration, overconfidence, or draw handling;
- shows fold-level improvement across multiple folds, not only one fixture cluster;
- has an interpretable mechanism and a small predeclared configuration;
- can be reproduced from committed scripts and reports without private data in Git.

A candidate can become production-discussion-ready only after:

- repeated offline validation or more seasons;
- shadow/live-style evaluation on future fixtures;
- a model decision record comparing champion, candidate, calibration, failure modes, and rollback risk.

No candidate should be promoted from one 109-fixture offline window alone.

## Recommended Implementation Sequence

1. Create a consolidated planning-aware runner, likely `scripts/run_champion_family_experiments.py`, or extend existing variant runners only if that keeps output clearer.
2. Add diagnostics to the variant wrapper outputs for resolved rho, train size, test size, home advantage, parameter ranges, and confidence/overconfidence.
3. Re-run a compact champion-family baseline set: `champion_dc_xg`, `dc_fit_rho_each_fold`, `txg_xg_pseudocount_010`, and `blend_dc_fit_txg_50_50`.
4. Add pseudocount sensitivity: `0.05`, `0.10`, `0.15`, `0.20`.
5. Add conservative decay sensitivity: `60`, `75`, `90`, with `45` as a diagnostic holdover if desired.
6. Add draw-calibration diagnostic rows, clearly separated as post-processing candidates.
7. Produce `reports/champion_family_experiments_summary.md` and `reports/champion_family_experiments_summary.json`.
8. Update this document only after results exist, preserving the predeclared grid and marking any deviations.

## Proposed Branch Deliverables

For this branch, target:

- `docs/champion_calibration_and_decay_plan.md`
- `scripts/run_champion_family_experiments.py`
- `reports/champion_family_experiments_summary.md`
- `reports/champion_family_experiments_summary.json`
- Optional diagnostic CSVs only if they are small, intentional, and not raw/private data:
  - fitted rho per fold;
  - pseudocount sensitivity by fold;
  - time-decay sensitivity by fold;
  - draw calibration metrics;
  - high-confidence misses.

## First Implementation Run

The first consolidated champion-family run is complete.

Implemented runner:

- `scripts/run_champion_family_experiments.py`

Generated artifacts:

- `reports/champion_family_experiments_summary.md`
- `reports/champion_family_experiments_summary.json`
- `reports/champion_family_experiments_fold_metrics.csv`
- `reports/champion_family_experiments_rho_diagnostics.csv`
- `reports/champion_family_experiments_draw_diagnostics.csv`

Run inputs and protocol:

- CSV: `data/exports/wsl_match_data_xg_phase3_20260626_fresh.csv`
- Evaluation window: 2025-10-01 to 2026-05-16
- Rolling folds: 19
- Evaluated fixtures per full candidate: 109
- xG columns verified: `home_np_xg`, `away_np_xg`, `home_xg`, `away_xg`

The current offline leaderboard is recorded in the report artifacts. No variant is promoted from this single offline window.

## What Not To Do

- Do not modify the operational champion implementation.
- Do not continue neural-network work on this branch.
- Do not add PyTorch, embeddings, or deeper models.
- Do not use market odds as training inputs or model features.
- Do not run a broad random hyperparameter search.
- Do not combine many small tweaks into opaque variants before isolating their effects.
- Do not promote a variant from one offline report.
- Do not hide draw degradation behind improved aggregate log loss.
- Do not mix market-only, matched-market, corrected NN, and champion-family results in one unlabeled leaderboard.
