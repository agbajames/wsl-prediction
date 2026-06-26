# WSL Model Improvement Roadmap Checkpoint

Date: 2026-06-26

This is a documentation-only checkpoint after the corrected xG Phase 3 neural-network experiment. It does not change the champion model, production prediction behaviour, API behaviour, dashboard behaviour, or experiment code.

## Executive Summary

The core model-improvement roadmap is now in a much clearer place. The project has moved from a single production xG/Dixon-Coles model to a repeatable evaluation framework with baselines, statistical challengers, feature-based ML challengers, market comparison, non-market blending, shadow-testing scaffolding, Monte Carlo simulation, and a rigorous neural-network audit.

The strongest tested probability-quality model on the common 109-fixture local comparison window is `dc_fit_rho_each_fold`, with Brier 0.5027 and log loss 0.8623. The most balanced champion-family candidate is `txg_xg_pseudocount_010`, with Brier 0.5038, log loss 0.8649, and accuracy 0.6330. The unchanged operational reference remains `champion_dc_xg`, with Brier 0.5052, log loss 0.8669, and accuracy 0.6239.

The corrected xG neural-network phase is complete and negative for promotion. The best NN was `nn_tiny_8_xg`, with mean log loss 0.9746, Brier 0.5711, and accuracy 0.5474. It did not beat `champion_dc_xg`, `improved_logistic_regression`, or the stronger champion-family/statistical candidates on probability quality. Neural-network work should be parked as research-only until the project has more seasons or richer timestamp-safe features.

Recommended next branch: `feature/champion-calibration-and-decay`.

That branch best continues the core model-improvement roadmap: take the already promising champion-family candidates, calibration diagnostics, draw-adjustment findings, and time-decay/xG weighting variants, then turn them into a small, predeclared, leakage-safe champion-family checkpoint. Do not widen neural networks, add PyTorch embeddings, or promote any model from one offline window.

## Repository Checkpoint

Verified repository context for this audit:

| item | status |
| --- | --- |
| Local path | `C:\Users\agbaj\projects\wsl-prediction` |
| Branch | `feature/rigorous-neural-network-challenger` |
| Remote | `https://github.com/agbajames/wsl-prediction.git` |
| Upstream state | Up to date with `origin/feature/rigorous-neural-network-challenger`, 0 ahead / 0 behind |
| Working tree | Not clean: untracked corrected NN report files are present |
| Protected champion files | `models/champion_dc_xg.py` and `model/wsl_xg_model.py` have no diff |

Generated/untracked NN artifacts observed in the working tree:

| path | note |
| --- | --- |
| `reports/neural_network_experiments_xg.md` | Corrected xG Phase 3 report; generated after fresh Supabase export |
| `reports/neural_network_experiments_xg_metadata.json` | Corrected xG Phase 3 metadata; generated after fresh Supabase export |

The fresh CSV export `data/exports/wsl_match_data_xg_phase3_20260626_fresh.csv` was used for the corrected NN run and is under the ignored exports area.

## Roadmap Evidence Inventory

| file | role in roadmap | key evidence |
| --- | --- | --- |
| `docs/model_evaluation_readiness_audit.md` | Original broad evaluation architecture audit | Identified the need for data contracts, champion freezing, shared backtesting, baselines, market benchmarking, simulation, and challenger model infrastructure. Warned that one WSL season is small for high-capacity ML. |
| `docs/model_roadmap_reconciliation.md` | Previous roadmap reconciliation | Rebuilt the roadmap into Phases 8A-8E after the first model-comparison and champion diagnostics work. Marked 8A and 8B as active/completed and NN as later research. |
| `docs/champion_model.md` | Champion definition | Defines `champion_dc_xg` as the frozen evaluation adapter around `model/wsl_xg_model.py`, with production logic kept separate. |
| `reports/model_comparison_first_run.md` | First shared model comparison | Established `champion_dc_xg` as the initial reference leader over 109 fixtures: Brier 0.5052, log loss 0.8669, accuracy 0.6239. |
| `docs/champion_diagnostics.md` and `reports/champion_diagnostics_first_run.md` | Champion error analysis | Confirmed champion remained reference; highlighted high-confidence misses, draw/favourite behaviour, and team-level diagnostics. |
| `docs/champion_calibration.md` and `reports/champion_calibration_first_run.md` | Calibration experiment | Temperature/shrinkage calibration worsened the trial split; original champion stayed reference. |
| `docs/champion_draw_adjustment.md` and `reports/champion_draw_adjustment_first_run.md` | Draw post-processing experiment | Fixed draw shrinkage slightly improved probability metrics in one artifact but reduced accuracy/actual-draw handling; keep as hypothesis only. |
| `docs/dixon_coles_variants.md`, `docs/time_decay_xg_variants.md`, and `reports/statistical_challengers_phase_8a_summary.md` | Phase 8A champion-family/statistical variants | `dc_fit_rho_each_fold` became best probability-quality candidate; `txg_xg_pseudocount_010` became the most balanced champion-family candidate. |
| `docs/regularised_team_strength.md` and `reports/regularised_team_strength_comparison_first_run.md` | Standalone statistical challenger | `regularised_team_strength` is the strongest standalone non-champion statistical challenger. |
| `docs/poisson_regression_challenger.md` and `reports/poisson_regression_comparison_first_run.md` | Poisson regression challenger | Useful transparent statistical challenger, but below regularised team strength and champion-family variants. |
| `docs/improved_logistic_regression.md`, `docs/tree_based_challenger.md`, and `reports/feature_ml_challengers_phase_8b_summary.md` | Phase 8B feature-based ML | `improved_logistic_regression` is best feature-ML model by Brier/log loss; `random_forest` best feature-ML accuracy but lower probability quality. |
| `docs/non_market_blending.md` and `reports/non_market_blending_first_run.md` | Phase 8D non-market blending | Best blend was `blend_dc_fit_txg_50_50`, but it did not beat `dc_fit_rho_each_fold`; useful candidate for shadow tracking, not promotion. |
| `docs/market_benchmark_strategy.md` and `reports/market_benchmark_2025_26.md` | Phase 1 market-implied benchmark | Market-implied no-vig benchmark scored 132 fixtures at Brier 0.4712, log loss 0.8127, accuracy 0.6515, subject to odds timing/licensing caveats. |
| `docs/model_vs_market_comparison.md` and `reports/model_vs_market_comparison_2025_26.md` | Phase 2 model-vs-market comparison | On 109 matched fixtures, market-implied benchmark beat `champion_dc_xg`: 0.4885/0.8369 vs 0.5052/0.8669. Market remains evaluation-only, not a model feature. |
| `docs/shadow_testing.md` and `evaluation/shadow.py` | Phase 8E shadow/live framework | Shadow prediction generation/replay scaffolding exists with default candidates, but real live use is blocked until approved future fixtures are available. |
| `docs/simulation_strategy.md`, `simulation/monte_carlo.py`, and `reports/monte_carlo_after_week_11_2025_26.md` | Monte Carlo simulation | Mid-season table simulation exists for decision-facing scenario analysis, separate from model selection. |
| `docs/neural_network_challenger_audit.md` | NN audit before rigorous run | Warned that NN is useful only as a controlled nonlinear diagnostic unless it beats probability metrics across folds/seeds. |
| `docs/neural_network_experiments.md` and `reports/neural_network_experiments_xg.md` | Corrected xG NN Phase 3 | Seven-architecture, three-seed, no-fallback xG experiment completed. Best NN did not beat champion or improved logistic; NN should be parked. |

## Completed Roadmap Phases

### Foundation and Champion Reference

Completed:

- Canonical Supabase-backed match data path through `data/supabase_client.py`.
- WSL match export workflow in `scripts/export_wsl_match_data.py`.
- Shared evaluation model protocol in `models/base.py`.
- Frozen champion adapter `models/champion_dc_xg.py` over the production implementation in `model/wsl_xg_model.py`.
- Rolling backtest framework in `evaluation/backtesting.py`.
- Metrics, calibration, comparison, diagnostics, and failure-analysis helpers in `evaluation/`.
- Model registry in `experiments/registry.py`.

Current registered models include:

- `champion_dc_xg`
- `naive_outcome_rate`
- `elo_baseline`
- `logistic_regression`
- `improved_logistic_regression`
- `neural_network`
- `regularised_team_strength`
- `poisson_regression`
- `random_forest`

### Phase 1: Market-Implied Benchmark

Completed as evaluation-only market work:

- `evaluation/market_benchmark.py`
- `scripts/run_market_benchmark.py`
- `docs/market_benchmark_strategy.md`
- `reports/market_benchmark_2025_26.md`

The market-implied benchmark scored 132 fixtures at Brier 0.4712, log loss 0.8127, and accuracy 0.6515. This is a strong external reference, but interpretation depends on odds provenance, snapshot timing, and licensing.

### Phase 2: Model vs Market Matched-Fixture Comparison

Completed as an offline comparison:

- `evaluation/model_market_comparison.py`
- `scripts/run_model_market_comparison.py`
- `docs/model_vs_market_comparison.md`
- `reports/model_vs_market_comparison_2025_26.md`

On 109 matched fixtures:

| model | Brier | log loss | accuracy |
| --- | --- | --- | --- |
| `market_implied_benchmark` | 0.4885 | 0.8369 | 0.6330 |
| `champion_dc_xg` | 0.5052 | 0.8669 | 0.6239 |
| `logistic_regression` | 0.5562 | 0.9560 | 0.5780 |
| `elo_baseline` | 0.5676 | 0.9597 | 0.5596 |
| `naive_outcome_rate` | 0.6401 | 1.0576 | 0.4587 |

This is not a market-blending model. Market odds remain out of model features and training.

### Phase 8A: Statistical and Champion-Family Challengers

Completed:

- Regularised team strength.
- Poisson regression.
- Dixon-Coles/champion-family variants.
- Time-decay and xG-weighting variants.
- Consolidated Phase 8A report.

Primary outcome:

| model | Brier | log loss | accuracy | interpretation |
| --- | --- | --- | --- | --- |
| `dc_fit_rho_each_fold` | 0.5027 | 0.8623 | 0.6147 | Best probability-quality candidate. |
| `txg_xg_pseudocount_010` | 0.5038 | 0.8649 | 0.6330 | Most balanced champion-family candidate. |
| `champion_dc_xg` | 0.5052 | 0.8669 | 0.6239 | Operational/reference model. |
| `regularised_team_strength` | 0.5256 | 0.9001 | 0.5963 | Strongest standalone non-champion statistical challenger. |
| `poisson_regression` | 0.5585 | 0.9426 | 0.5413 | Useful transparent challenger, but not leading. |

Phase 8A created the strongest future direction: improve and validate champion-family variants before production promotion.

### Phase 8B: Feature-Based ML Challengers

Completed:

- Improved logistic regression with feature-group ablations.
- Random forest challenger.
- Feature importance artifact.
- Consolidated Phase 8B report.

Primary outcome:

| model | Brier | log loss | accuracy | interpretation |
| --- | --- | --- | --- | --- |
| `improved_logistic_regression` | 0.5384 | 0.9356 | 0.5780 | Best feature-ML model by probability quality. |
| `random_forest` | 0.5491 | 0.9385 | 0.5872 | Best feature-ML accuracy, worse probability quality than improved logistic. |
| `logistic_regression` | 0.5562 | 0.9560 | 0.5780 | Original ML baseline. |

Feature-rich models still trail the statistical/champion-family leaders. The improved logistic `xg` feature group was best; fuller feature sets underperformed, which is a warning against adding feature volume without more data.

### Phase 8C: Neural Network Challenger

Completed:

- NumPy neural-network architecture ladder.
- Seven predeclared architectures: `[]`, `[8]`, `[16]`, `[32]`, `[64]`, `[32, 16]`, `[64, 32]`.
- Three seeds: 42, 7, 123.
- Time-aware validation/early stopping.
- Per-seed, per-fold, prediction, training-history, and metadata outputs.
- Corrected real-xG run from fresh Supabase export with zero xG missingness and no fallback-to-goals.

Corrected xG Phase 3 result:

| item | value |
| --- | --- |
| CSV | `data/exports/wsl_match_data_xg_phase3_20260626_fresh.csv` |
| Rows | 132 |
| Date range | 2025-09-05 to 2026-05-16 |
| xG columns | `home_np_xg`, `away_np_xg`, `home_xg`, `away_xg` |
| xG missingness | 0 for all four xG columns |
| Evaluation fixtures | 109 |
| Best NN | `nn_tiny_8_xg` |
| Best NN mean log loss | 0.9746 |
| Best NN mean Brier | 0.5711 |
| Best NN mean accuracy | 0.5474 |

Conclusion: park neural networks. Do not scale to larger hidden layers, PyTorch embeddings, or deeper models yet. Revisit only after adding more seasons or richer timestamp-safe pre-match features.

### Phase 8D: Non-Market Blending

Completed as an offline fixed-blend experiment:

| model | Brier | log loss | accuracy | interpretation |
| --- | --- | --- | --- | --- |
| `dc_fit_rho_each_fold` | 0.5027 | 0.8623 | 0.6147 | Best ranked row. |
| `blend_dc_fit_txg_50_50` | 0.5031 | 0.8634 | 0.6239 | Best fixed blend. |
| `txg_xg_pseudocount_010` | 0.5038 | 0.8649 | 0.6330 | Strong balanced candidate. |
| `champion_dc_xg` | 0.5052 | 0.8669 | 0.6239 | Operational/reference model. |

Blending is promising as evaluation context and shadow tracking, but no blend should be promoted from one offline window.

### Phase 8E: Shadow/Live Prediction Framework

Prepared:

- `evaluation/shadow.py`
- `scripts/generate_shadow_predictions.py`
- `scripts/evaluate_shadow_predictions.py`
- `scripts/normalise_shadow_fixtures.py`
- `docs/shadow_testing.md`

Default shadow candidates:

- `champion_dc_xg`
- `dc_fit_rho_each_fold`
- `txg_xg_pseudocount_010`
- `blend_dc_fit_txg_50_50`
- `regularised_team_strength`
- `improved_logistic_regression`
- `random_forest`

Blocked: true live/shadow prediction generation needs real approved upcoming fixtures. At season end, there are no future fixtures to predict, so real shadow evaluation must wait for the next schedule or approved future fixture file.

### Monte Carlo Simulation

Completed:

- `simulation/monte_carlo.py`
- `scripts/run_monte_carlo_simulation.py`
- `docs/simulation_strategy.md`
- `reports/monte_carlo_after_week_11_2025_26.md`

The current simulation report uses 10,000 simulations after Matchweek 11. It is decision-facing scenario analysis, not model selection. It uses scoreline simulation from model expected-goals lambdas and tracks table outcomes such as title/top-3/top-4 probabilities.

## Current Leaderboards

Do not mix these leaderboards without the labels. The common model-comparison/challenger reports cover 109 evaluated fixtures. The market-only report covers 132 fixtures. The model-vs-market report covers 109 matched fixtures. The corrected NN Phase 3 report uses the same 109-fixture evaluation window but a fresh Supabase xG export and multi-seed NN aggregation.

### Common 109-Fixture Model and Challenger Window

Ranked by log loss, then Brier, then accuracy:

| rank | model | log loss | Brier | accuracy | label |
| --- | --- | --- | --- | --- | --- |
| 1 | `dc_fit_rho_each_fold` | 0.8623 | 0.5027 | 0.6147 | Best probability-quality champion-family candidate |
| 2 | `blend_dc_fit_txg_50_50` | 0.8634 | 0.5031 | 0.6239 | Best fixed non-market blend; included separately because it comes from the blend experiment |
| 3 | `txg_xg_pseudocount_010` | 0.8649 | 0.5038 | 0.6330 | Most balanced champion-family candidate |
| 4 | `champion_dc_xg` | 0.8669 | 0.5052 | 0.6239 | Operational/reference model |
| 5 | `regularised_team_strength` | 0.9001 | 0.5256 | 0.5963 | Best standalone non-champion statistical challenger |
| 6 | `improved_logistic_regression` | 0.9356 | 0.5384 | 0.5780 | Best feature-ML model by probability quality |
| 7 | `random_forest` | 0.9385 | 0.5491 | 0.5872 | Best feature-ML accuracy |
| 8 | `poisson_regression` | 0.9426 | 0.5585 | 0.5413 | Transparent statistical challenger |
| 9 | `logistic_regression` | 0.9560 | 0.5562 | 0.5780 | Original ML baseline |
| 10 | `elo_baseline` | 0.9596 | 0.5676 | 0.5596 | Simple rating baseline |
| 11 | `neural_network` proof of concept | 0.9762 | 0.5567 | 0.6055 | Older single-config NN POC, not corrected Phase 3 |
| 12 | `nn_tiny_8_xg` corrected Phase 3 | 0.9746 | 0.5711 | 0.5474 | Best corrected xG NN across 3 seeds |
| 13 | `naive_outcome_rate` | 1.0576 | 0.6401 | 0.4587 | Sanity baseline |

Notes:

- `blend_dc_fit_txg_50_50` comes from the non-market blending experiment and should be considered a carry-forward candidate, not a standalone model family.
- The older `neural_network` proof-of-concept and corrected `nn_tiny_8_xg` are both shown for context. The corrected Phase 3 result supersedes the earlier NN diagnostic for the NN roadmap decision.
- The operational champion remains `champion_dc_xg` despite small offline improvements from champion-family variants.

### Matched Model vs Market Window

On 109 matched fixtures:

| rank | model | log loss | Brier | accuracy |
| --- | --- | --- | --- | --- |
| 1 | `market_implied_benchmark` | 0.8369 | 0.4885 | 0.6330 |
| 2 | `champion_dc_xg` | 0.8669 | 0.5052 | 0.6239 |
| 3 | `logistic_regression` | 0.9560 | 0.5562 | 0.5780 |
| 4 | `elo_baseline` | 0.9597 | 0.5676 | 0.5596 |
| 5 | `naive_outcome_rate` | 1.0576 | 0.6401 | 0.4587 |

The market result is important but not directly a production model candidate. It is an external benchmark and future calibration/edge-analysis reference.

### Market-Only Full Odds Window

On 132 market fixtures:

| model | log loss | Brier | accuracy |
| --- | --- | --- | --- |
| `market_implied_benchmark` | 0.8127 | 0.4712 | 0.6515 |

This confirms the market reference is strong across the full odds file, subject to source/timing/licensing constraints.

## Strategic Interpretation

### What Is Currently Strongest?

The strongest tested probability-quality candidate is `dc_fit_rho_each_fold`. It slightly beats the unchanged champion on the common offline window.

The strongest operational reference remains `champion_dc_xg`. It is stable, frozen, integrated, and still extremely close to the best offline variants.

### Is the Champion Still the Right Reference?

Yes. `champion_dc_xg` should remain the reference until a candidate wins across repeated leakage-safe evaluations and preferably shadow/live-style validation. The current champion-family improvements are small and promising, not production-proof.

### Did Any Challenger Beat the Champion on Probability Quality?

Yes, but narrowly and only offline:

- `dc_fit_rho_each_fold` improved Brier and log loss.
- `txg_xg_pseudocount_010` improved Brier, log loss, and accuracy.
- `blend_dc_fit_txg_50_50` beat the champion but did not beat `dc_fit_rho_each_fold`.

No feature-ML, random forest, Poisson regression, or neural-network challenger beat the champion on probability quality.

### Did Neural Networks Add Value?

No for current model selection. The rigorous corrected xG phase showed that the best tiny NN still trailed the champion and improved logistic regression by log loss and Brier score. Wider/deeper architectures did not help and showed mild overfit/volatility signals.

Neural-network work is now sufficiently tested for the current dataset. Park it as research-only.

### What Should Be Parked?

Park:

- Larger MLP grids.
- PyTorch rewrites.
- Team embeddings.
- Deep architectures.
- Accuracy-only NN promotion arguments.
- Any NN continuation on the same single-season dataset without new data or new timestamp-safe features.

### What Deserves Continuation?

Continue:

- Champion-family variants: fitted rho, xG pseudocounts, time decay, shrinkage, penalty handling, home advantage handling.
- Calibration/draw diagnostics, but as fold-safe offline experiments.
- Non-market blending of the best champion-family candidates.
- Market comparison and edge analysis, if odds provenance/timing is acceptable.
- Shadow/live framework preparation and first real run when future fixtures exist.

### Current Blockers

| blocker | impact |
| --- | --- |
| No future fixtures at season end | Blocks true live/shadow prediction generation and replay. |
| One WSL season of xG-backed data | Limits high-capacity ML, neural nets, and broad hyperparameter search. |
| Odds source/timing/licensing uncertainty | Limits strength of market conclusions and public reporting. |
| No stable match/team IDs in the core schema | Keeps matching string/date-based and more fragile. |
| Small differences among champion-family variants | Requires repeated validation before promotion. |

## Forward Roadmap

### A. Immediate Cleanup and Checkpoint

Do next on the current NN branch:

- Preserve `reports/neural_network_experiments_xg.md` and `reports/neural_network_experiments_xg_metadata.json` intentionally if the branch is meant to retain the corrected xG result.
- Decide whether generated CSV/large CSV artifacts remain ignored local evidence or need a documented reproduction path only.
- Document the NN conclusion as parked/research-only.
- Avoid committing secrets, private exports, or large ignored data.
- Keep `models/champion_dc_xg.py` and `model/wsl_xg_model.py` untouched.

### B. Champion-Family Improvement Branch

This is the best next core model-improvement branch.

Use a small predeclared experiment plan around existing evidence:

- Fit Dixon-Coles `rho` inside each fold, with stability diagnostics.
- Tune xG pseudocounts around the already promising `txg_xg_pseudocount_010`.
- Recheck time decay and recency weighting with conservative grids.
- Test caps/shrinkage on attack/defence strengths.
- Review penalty xG and home/away penalty rates.
- Review home advantage handling.
- Explore promoted/new team priors only if timestamp-safe.
- Revisit draw calibration and probability calibration as post-processing candidates.
- Test fixed blends only among the strongest champion-family candidates.

Guardrails:

- Predeclare the grid.
- Keep champion implementation frozen.
- Compare on identical folds.
- Rank by log loss, then Brier, then accuracy.
- Treat one-window wins as hypotheses until shadow/live validation.

### C. Market Comparison and Calibration Branch

Continue after or alongside champion-family work:

- Re-run matched model-vs-market with the latest carry-forward candidates, not only the initial `champion_dc_xg`/logistic/Elo/naive set.
- Keep market odds out of model features unless a separate market-informed branch is explicitly created.
- Maintain proportional no-vig normalization from raw odds.
- Add edge analysis: when model and market disagree, where does each win by row log loss?
- Add calibration and overconfidence diagnostics against the market reference.
- Track draw pricing and underdog/favourite failures.
- Document odds timestamp/source limitations prominently.

### D. Shadow/Live Prediction Branch

True live/shadow generation is blocked until future fixtures or the next season schedule exists.

Prepare now:

- Keep fixture normalization scripts ready.
- Keep default shadow model list focused on champion, champion-family candidates, best blend, regularised team strength, improved logistic, and random forest.
- Prepare a runbook for the first 2026-27 fixture file.
- Confirm generated shadow artifacts include git SHA, timestamp, model config, fixture IDs, and prediction IDs.
- Do not fabricate upcoming fixtures.

### E. Data Expansion Branch

Required before revisiting high-capacity ML or NN work:

- Add more WSL seasons with goals and xG.
- Add canonical `season`, `competition`, `match_id`, and stable `team_id`.
- Add kickoff timestamps and prediction timestamps.
- Add timestamp-safe pre-match context: rest days, fixture congestion, travel if available, promoted-team priors, squad availability, injuries, lineups only if available pre-match, and squad value only if timestamped historically.
- Add odds snapshot metadata if market work becomes central.

Revisit neural networks only after this data layer is materially richer.

## Recommended Next Branch

Recommended branch:

```text
feature/champion-calibration-and-decay
```

Why this branch:

- It continues the strongest current evidence rather than chasing the now-negative NN path.
- It can build directly on `dc_fit_rho_each_fold`, `txg_xg_pseudocount_010`, time-decay/xG variants, calibration diagnostics, and draw-adjustment work.
- It keeps the work close to the existing champion model family, which is structurally well matched to small WSL samples.
- It can produce practical promotion candidates for later shadow/live testing without modifying the champion implementation.

Suggested first deliverable on that branch:

- A predeclared champion-family experiment plan and runner/report update that compares fitted rho, xG pseudocount, conservative time decay, draw calibration, and fixed champion-family blends on identical folds.

## Concrete Next Steps

1. Decide what to commit from the current NN branch: at minimum, the corrected xG NN markdown and metadata report if the branch is meant to preserve audit evidence.
2. Add this checkpoint report to the documentation set.
3. Open `feature/champion-calibration-and-decay`.
4. Predeclare the small champion-family grid before running new experiments.
5. Re-run only the targeted champion-family/calibration/decay experiments on the same folds.
6. Produce a season-end model decision candidate report that separates operational champion, best offline candidate, and shadow-ready candidates.
7. When future fixtures exist, run the prepared shadow workflow with `champion_dc_xg`, `dc_fit_rho_each_fold`, `txg_xg_pseudocount_010`, `blend_dc_fit_txg_50_50`, `regularised_team_strength`, `improved_logistic_regression`, and `random_forest`.

## Risks and Blockers

- The observed champion-family gains are small and may not survive new fixtures.
- Repeated offline variant testing can overfit the same 109-fixture window.
- Market odds are strong but require careful provenance and timing controls.
- Shadow/live validation cannot proceed without future fixtures.
- High-capacity ML remains underpowered on one season.
- Generated data/report artifacts need intentional handling to avoid accidental clutter or private-data commits.

## What Not To Do Next

- Do not modify `model/wsl_xg_model.py` or `models/champion_dc_xg.py` as part of roadmap work.
- Do not promote an NN based on accuracy.
- Do not expand to larger neural architectures, PyTorch, or team embeddings on the current data.
- Do not mix market-only, matched-market, fallback-goals diagnostics, and corrected xG runs in one unlabeled leaderboard.
- Do not fabricate shadow fixtures.
- Do not use market odds as features unless a separate market-informed experiment is explicitly designed.
- Do not commit secrets, `.env`, private odds exports, or raw fixture exports.
