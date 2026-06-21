# Model Roadmap Reconciliation

This document reconciles the original model-evaluation roadmap with the repository state after the first real comparison and the first champion-focused offline experiments.

The main conclusion is simple: `champion_dc_xg` remains the best tested model so far, but the modelling roadmap is not complete. The project has built strong evaluation infrastructure and tested only the first challenger tier plus several champion post-processing hypotheses. The next work should broaden the challenger suite before any production model decision.

## Current State

The repository now has a usable local champion-vs-challenger evaluation path:

- Canonical model/data contract documentation in `docs/data_contract.md`.
- A frozen champion adapter in `models/champion_dc_xg.py` backed by `model/wsl_xg_model.py`.
- A shared model protocol in `models/base.py`.
- Rolling backtests in `evaluation/backtesting.py`.
- Metrics, calibration, comparison, and failure-analysis helpers in `evaluation/`.
- Registered local models in `experiments/registry.py`: `champion_dc_xg`, `naive_outcome_rate`, `elo_baseline`, and `logistic_regression`.
- Model comparison runner in `scripts/run_model_comparison.py`.
- First real model-comparison artefacts in `reports/model_comparison_first_run.md` and `reports/model_comparison_first_run.json`.
- Champion diagnostics and post-processing experiment reports through calibration and draw adjustment on current `main`.

Current measured reference result from `reports/model_comparison_first_run.md`:

| rank | model | Brier | log loss | accuracy |
| --- | --- | --- | --- | --- |
| 1 | `champion_dc_xg` | 0.5052 | 0.8669 | 0.6239 |
| 2 | `logistic_regression` | 0.5562 | 0.9560 | 0.5780 |
| 3 | `elo_baseline` | 0.5676 | 0.9596 | 0.5596 |
| 4 | `naive_outcome_rate` | 0.6401 | 1.0576 | 0.4587 |

The unadjusted `champion_dc_xg` remains the current reference model. That is a current evidence statement, not a final roadmap decision.

## Implemented Models And Experiments

| model/experiment | file(s) | status | changes production behaviour? | evaluated? | key result |
| --- | --- | --- | --- | --- | --- |
| `champion_dc_xg` | `model/wsl_xg_model.py`, `models/champion_dc_xg.py`, `experiments/configs/champion_dc_xg.yaml`, `tests/test_champion_model_adapter.py` | Implemented on main; frozen adapter around existing production model | No adapter behaviour change intended; production model remains in `model/wsl_xg_model.py` | Yes | Best tested model in first comparison: Brier 0.5052, log loss 0.8669, accuracy 0.6239 |
| `naive_outcome_rate` | `models/baselines.py`, `experiments/configs/naive_baseline.yaml`, `tests/test_baseline_models.py` | Implemented on main | No | Yes | Sanity baseline; weakest first-run model with Brier 0.6401, log loss 1.0576, accuracy 0.4587 |
| `elo_baseline` | `models/baselines.py`, `experiments/configs/elo_baseline.yaml`, `tests/test_baseline_models.py` | Implemented on main | No | Yes | Interpretable results-based baseline; Brier 0.5676, log loss 0.9596, accuracy 0.5596 |
| `logistic_regression` | `models/logistic.py`, `features/team_form.py`, `experiments/configs/logistic_regression.yaml`, `tests/test_logistic_model.py` | Implemented on main | No | Yes | Best challenger so far but below champion: Brier 0.5562, log loss 0.9560, accuracy 0.5780 |
| Integrated model comparison | `scripts/run_model_comparison.py`, `evaluation/compare.py`, `evaluation/calibration.py`, `evaluation/failure_analysis.py`, `tests/test_model_comparison_runner.py` | Implemented on main | No | Yes | Produced `reports/model_comparison_first_run.*` over 109 matches per model |
| Champion diagnostics | `evaluation/diagnostics.py`, `scripts/run_champion_diagnostics.py`, `docs/champion_diagnostics.md`, `reports/champion_diagnostics_first_run.*` | Implemented on main | No | Yes | Confirmed champion remains reference; highlighted high-confidence misses, draw/favourite behaviour, and team-level patterns |
| Champion calibration experiment | `evaluation/calibrators.py`, `scripts/run_champion_calibration_experiment.py`, `docs/champion_calibration.md`, `reports/champion_calibration_first_run.*` | Implemented on main | No | Yes | Temperature plus shrinkage calibration worsened trial Brier/log loss and reduced accuracy; do not replace champion |
| Champion draw-adjustment experiment | `evaluation/draw_adjustment.py`, `scripts/run_champion_draw_adjustment_experiment.py`, `docs/champion_draw_adjustment.md`, `reports/champion_draw_adjustment_first_run.*` | Implemented on main | No | Yes | Best fixed draw shrinkage slightly improved Brier/log loss but reduced accuracy and worsened actual-draw handling; keep champion reference |
| Champion favourite-shrinkage experiment | `evaluation/favourite_shrinkage.py`, `scripts/run_champion_favourite_shrinkage_experiment.py`, `docs/champion_favourite_shrinkage.md`, `reports/champion_favourite_shrinkage_first_run.*` | Pushed on `origin/codex/champion-favourite-shrinkage-experiment`; not present on current `main` at reconciliation time | No | Yes in pushed branch | No tested favourite-shrinkage variant improved Brier or log loss; original champion remained best tested option |

## Remaining Model Candidates

The original audit in `docs/model_evaluation_readiness_audit.md` explicitly named several remaining families: gradient boosting, neural networks, market-only baselines, model-plus-market blends, and PL-to-WSL transfer/adaptation. The statistical variants below are partly inferred from the current champion family, diagnostics, and the need to avoid stopping after only one ML challenger.

| candidate | source | what it is | why test it | data requirements | overfitting risk on one WSL season | timing | priority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Regularised attack/defence team-strength model | Inferred from champion family and audit architecture | A simpler statistical model estimating team attack/defence strengths with stronger shrinkage/priors than the current xG/DC setup | Tests whether a more regularised model beats the current champion on small samples | Match outcomes, xG if included, stable teams, fold-aware fitting | Medium; regularisation helps but team-level parameters are still noisy | Short-term | High |
| Poisson regression challenger | Inferred from champion Poisson structure | Regression model for home/away goals or xG-derived goal rates, then 1X2 probabilities from scoreline simulations | Gives a transparent statistical challenger between Elo/logistic and the full champion | Goals, xG optional, home/away/team features | Medium | Short-term | High |
| Dixon-Coles parameter variants | Inferred from `champion_dc_xg` and `docs/model_evaluation_readiness_audit.md` champion definition | Compare fixed/fitted `rho`, goal truncation, penalty handling, and xG shrinkage variants without rewriting champion | Tests whether champion configuration, not model family, is leaving performance on the table | Same as champion xG schema | Medium-high if too many variants are searched | Short-term | High |
| Time-decay variants | Inferred from frozen champion config and roadmap risks | Test alternate half-lives or recency weighting for team-strength estimates | WSL team strength can move quickly; current fixed decay may be suboptimal | Dated historical matches and xG | Medium-high on one season | Short-term | Medium-high |
| Form-weighted xG challenger | Inferred from champion diagnostics and logistic feature work | Blend team xG strength with recent form/xG trend features | Directly tests whether recent xG form improves the champion without jumping to complex ML | Match dates, xG, rolling feature builder | High unless feature windows are predeclared | Medium-term | Medium |
| Improved logistic regression | Explicitly supported by `models/logistic.py`; expansion inferred | Add richer leakage-safe features, stronger regularisation choices, and ablation reporting | Current logistic is the best challenger; improving it is a natural low-risk next ML step | Results, team-form features, optional xG/form features | Medium | Medium-term | High |
| Random forest / gradient boosting challenger | Explicit in original audit as LightGBM/XGBoost; random forest inferred as adjacent tree baseline | Tree-based tabular challenger using rolling team/form/xG features | Captures nonlinear interactions that logistic regression cannot | More seasons strongly preferred, leakage-safe features | High on one WSL season | Medium-term | Medium |
| Neural network challenger | Explicit in original audit | Simple tabular MLP proof-of-concept only | Useful research benchmark, but unlikely to be robust on current sample alone | Multiple seasons or strong regularisation; strict validation | Very high | Later | Low-medium |
| Ensemble/blended model | Explicit as model-plus-market blend; champion/logistic/Elo blend inferred | Weighted blend of champion, logistic, Elo, and/or other model probabilities fitted inside training folds | Ensembles can improve probability quality without replacing interpretable components | Aligned out-of-fold predictions, actuals; enough folds | Medium if weights are learned carefully | Medium-term | High |
| Market-only baseline | Explicit in original audit; blocked on odds data | Convert bookmaker odds to H/D/A probabilities and evaluate as a model | Market is often the strongest external benchmark | Odds snapshots with timestamps before kickoff | Medium; data timing/liquidity matters | Later or data-dependent | High if data exists |
| Market-blended model | Explicit in original audit | Blend model probabilities with market probabilities | Tests whether model adds signal beyond market or benefits from market information | Aligned model predictions, market odds, actuals, snapshot timing | Medium | Later or data-dependent | High if data exists |
| Historical/multi-season transfer-learning approach | Explicit as PL-to-WSL transfer/adaptation | Train/adapt across additional seasons or competitions, including league indicators/calibration | Addresses the biggest limitation: one WSL season is too small for robust selection | Multi-season WSL and/or external league match/xG data, licensing clarity | Medium-high due domain shift | Later/data-dependent | Medium |

## Recommended Next Phases

The original audit put market blending and transfer work immediately after the first baselines. The corrected roadmap should first fill the statistical and feature-based challenger gap, because the current evidence suite has only tested naive, Elo, and one logistic model against the champion.

### Phase 8A - Regularised Statistical Challenger Models

- Implement a regularised attack/defence strength challenger.
- Implement a Poisson regression challenger.
- Run Dixon-Coles parameter/configuration comparisons as offline variants of the champion family.
- Predeclare small grids for shrinkage, decay, `rho`, and xG weighting to avoid overfitting.
- Output: `reports/statistical_challengers_*.md` and JSON summaries.

### Phase 8B - Feature-Based Machine Learning Challengers

- Improve the existing logistic regression challenger with predeclared feature additions and ablation.
- Add a conservative tree baseline, starting with random forest or gradient boosting only if dependencies and sample-size limits are acceptable.
- Include feature importance and leakage checks.
- Output: model comparison report against the unchanged `champion_dc_xg`.

### Phase 8C - Neural Network Proof Of Concept

- Add a simple MLP over tabular features only after the statistical and tree baselines exist.
- Keep validation leakage-aware and report sample-size limitations prominently.
- Treat this as research, not a production candidate, unless more historical data becomes available.

### Phase 8D - Ensemble And Blending Experiments

- Test champion plus logistic blend.
- Test champion plus Elo blend.
- Test champion plus statistical challenger blend if Phase 8A produces a strong challenger.
- Add market blend only if odds snapshots exist with reliable pre-match timestamps.
- Learn blend weights only inside training folds.

### Phase 8E - Shadow Testing On Upcoming Matches

- Generate predictions before matches are played.
- Persist prediction artefacts with model name, version, git SHA, fixture keys, and timestamps.
- Replay after results are known.
- Compare live/shadow performance to backtest results and report drift.

### Phase 9 - Production Evaluation Pack And Model Decision Record

Do this only after the expanded challenger suite and at least one shadow/live-style evaluation run.

- Produce a model decision record.
- Include champion/challenger metrics, calibration, failure analysis, stability, operational risks, and rollback plan.
- Decide whether to keep `champion_dc_xg`, promote a challenger, or run a shadow candidate longer.

## Checklist

Already done on current `main`:

- [x] Data contract and schema baseline.
- [x] Champion adapter and frozen champion config.
- [x] Rolling backtest framework.
- [x] Evaluation reporting framework.
- [x] Naive and Elo baselines.
- [x] Logistic regression challenger.
- [x] Integrated model comparison runner.
- [x] WSL match-data export script.
- [x] First real model comparison report.
- [x] Champion diagnostics report.
- [x] Champion calibration experiment.
- [x] Champion draw-adjustment experiment.

Already done in pushed branch, not yet present on current `main` at reconciliation time:

- [x] Champion favourite-shrinkage experiment: `origin/codex/champion-favourite-shrinkage-experiment`.

Active next:

- [x] Phase 8A regularised statistical challengers.
- [x] Poisson regression challenger.
- [x] Dixon-Coles parameter/configuration comparison.
- [x] Time-decay and xG-weighting variants with predeclared grids.
- [x] Consolidated Phase 8A statistical challengers report.

Later:

- [ ] Phase 8B improved logistic and tree challengers.
- [ ] Phase 8C neural network proof-of-concept.
- [ ] Phase 8D non-market ensemble/blending.
- [ ] Phase 8E shadow testing on upcoming matches.
- [ ] Phase 9 production evaluation pack and model decision record.

Blocked by data:

- [ ] Market-only baseline and market blend until pre-match odds snapshots exist and are licensed/usable.
- [ ] Historical/multi-season transfer learning until additional WSL or external-league data is available with clear licensing.
- [ ] Robust high-capacity ML model selection until the sample expands beyond one WSL season or repeated shadow runs.

## Validation Note

This reconciliation is documentation-only. It does not implement a new model and does not change production, API, dashboard, or prediction behaviour.
