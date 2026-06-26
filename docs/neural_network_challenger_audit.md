# Neural Network Challenger Audit

## Executive Summary

The current `neural_network` model is a useful proof of concept, not yet a rigorous challenger to `champion_dc_xg`. It is implemented as a dependency-light NumPy MLP in `models/neural_network.py`, registered through `experiments/registry.py`, and evaluated on the same rolling backtest path as the other local challengers. It uses the same leakage-aware rolling xG/team-form feature builder as the improved logistic model, scales features inside each training fold, and emits 1X2 probabilities through a softmax output layer.

The honest conclusion is that the present WSL dataset is small for neural networks. The first evaluated run covers 109 test fixtures over 19 weekly time-aware folds, with training folds growing from 23 to 125 fixtures. That is enough to test a tiny regularised network as a research challenger, but not enough to justify a deep or high-capacity model. Existing results support this caution: `neural_network` reached 0.6055 accuracy, but its Brier score was 0.5567 and log loss was 0.9762, materially behind `champion_dc_xg` at 0.5052 Brier and 0.8669 log loss. Its worst misses were highly confident, which is exactly the failure mode that matters for probability forecasting.

The strongest honest case for a neural network challenger is not "it will beat the champion now." It is that a tightly controlled neural phase can test whether nonlinear interactions in rolling xG/form features add anything beyond logistic regression, trees, and regularised statistical models. The likely breaking point is calibration and generalisation: there are too few fixtures and too few repeated team-state examples to support much capacity.

The champion model must remain untouched. Neural-network work should remain a separate challenger track with explicit go/no-go criteria, strict time-aware validation, and probability-quality metrics as the selection target.

## Current Implementation Map

| Area | Current state |
| --- | --- |
| Model implementation | `models/neural_network.py`, class `NeuralNetworkChallenger` |
| Registry path | Registered as `neural_network` in `experiments/registry.py` |
| Config | `experiments/configs/neural_network.yaml` |
| Tests | `tests/test_neural_network_model.py` |
| Evaluation path | `scripts/run_model_comparison.py` -> `evaluation/backtesting.py` -> shared report helpers |
| Existing report | `reports/neural_network_proof_of_concept_first_run.md/json` |
| Champion reference | `models/champion_dc_xg.py` adapter over `model/wsl_xg_model.py`; not modified by NN work |

The backtest framework is date-aware. `evaluation/backtesting.py` builds folds where training rows are strictly before the test window. In the first neural-network report the test period was 2025-10-01 to 2026-05-16, with 19 rolling folds and 109 evaluated fixtures per model.

## Current Neural Network Audit

| Question | Answer |
| --- | --- |
| Where is it implemented? | `models/neural_network.py` as `NeuralNetworkChallenger`. |
| What features does it use? | Default `feature_group="xg"` from `ImprovedTeamFormFeatureBuilder`: base team-form rates plus rolling/cumulative xG-for/xG-against differentials. |
| How are features preprocessed? | Features are built sequentially for training rows, then standardised with mean/std fitted on the training fold only. Zero std is replaced with 1.0. |
| How are categorical variables handled? | Team names are not one-hot encoded, target encoded, or embedded. They are used only to maintain rolling team summaries. |
| How are missing values handled? | Played rows missing teams/goals are dropped by the feature builder. xG uses `home_np_xg`/`away_np_xg`, falling back to total xG, then goals if xG is missing. This is practical but should be logged because fallback-to-goals changes the feature meaning. |
| What target is it predicting? | Three-class match result: home win `H`, draw `D`, away win `A`, derived from final goals. |
| Is it predicting 1X2 probabilities directly? | Yes. It trains a softmax classifier directly over H/D/A. |
| Does it output calibrated probabilities? | It outputs softmax probabilities, but there is no calibration layer or validation-based recalibration. Softmax is not the same as calibrated probability. |
| Architecture | One hidden-layer MLP. |
| Hidden layers/units | One hidden layer, default 8 units. |
| Activation | `tanh`. |
| Optimiser/training method | Full-batch gradient descent implemented manually in NumPy. |
| Loss function | Multiclass cross-entropy gradient via softmax error. Training loss is not logged. |
| L2 regularisation | Yes, default `l2_penalty=0.01` on hidden and output weights. Biases are not regularised. |
| Dropout | No. |
| Early stopping | No. It always runs `max_iter`, default 500. |
| Feature scaling | Yes, training-fold standardisation. |
| Train/test split time-aware? | Yes in the shared rolling backtest. |
| Leakage risk | Mostly controlled in feature construction and fold splitting. Remaining risks are fallback-to-goals for missing xG, any future-derived source columns before export, and hyperparameter selection on the same final window if repeated manually. |
| Evaluation consistency | Yes for local model comparison against `champion_dc_xg` and challengers. Market comparison is separate and only possible on matched odds fixtures. |
| Random seeds controlled? | Initial weights use `np.random.default_rng(random_seed)`, default 42. |
| Reproducible? | Mostly yes for fixed data/order/NumPy version. There is no multi-seed reporting, so stability is unknown. |
| Metrics logged clearly? | Final comparison, calibration bins, confidence buckets, failure analysis, and favourite breakdown are logged. Training/validation loss, fold-by-fold NN diagnostics, parameter counts, and seed variance are not logged. |

## Current Weaknesses

- The model has no validation split inside each training fold, so it cannot early-stop, tune capacity, or report train-validation divergence.
- It uses only one fixed architecture. There is no logistic/no-hidden-layer sanity baseline in the same NN experiment harness.
- It has a fixed single seed. One seed can make a small neural network look better or worse by chance.
- It emits softmax probabilities without calibration checks by class. Existing aggregate calibration tables are useful but not enough for a serious NN audit.
- It does not report train loss, validation loss, parameter count, gradient stability, or per-fold metrics.
- It does not test whether the neural network is learning more than the improved logistic feature set. The right first question is whether a no-hidden-layer baseline behaves sensibly.
- It can be overconfident. In the first report, several NN misses assigned less than 5% to the actual outcome, including row log losses above 3.0.

## Data Suitability Assessment

The current evidence window is small. The first comparison uses one WSL season window with 109 evaluated fixtures and training folds from 23 to 125 fixtures. WSL has a small league size, so each team contributes relatively few home/away examples, and promoted/new teams have sparse histories early in the season.

This is not enough for a deep neural network to generalise reliably. It is enough for:

- A multinomial logistic baseline over leakage-safe engineered features.
- Tiny MLPs with strong L2 and early stopping.
- A small, predeclared architecture ladder.
- Diagnostics about overconfidence, draw handling, favourite bias, and nonlinear interactions.

It is not enough for:

- Large hidden layers selected by a broad grid search.
- Multiple stacked layers without strong evidence from smaller models.
- Team embeddings as a promotion candidate.
- Feature-rich hybrid models that include many sparse categorical or context features.

Simpler models are likely to dominate on probability quality until there are more seasons, more timestamp-safe features, or repeated shadow/live evaluations. The champion and champion-family variants already encode football structure through xG-driven team attack/defence and scoreline probabilities. A neural network has to overcome both data scarcity and the lack of explicit football scoreline structure.

## Feature Audit

Current NN default features are pre-match safe if the source rows are themselves timestamp-safe:

- Cumulative points per match, goals for/against, and differences.
- Cumulative xG for/against and xG differential.
- Recent-window xG differential using only previous rows.
- An implicit home feature via `home_indicator`.
- Match counts and team history are represented only when feature groups include form/full; default xG group includes base plus xG features.

Suspicious or improvement areas:

- xG fallback to goals should be surfaced in diagnostics. Goals are known after historical matches, so using past goals is safe, but mixed xG/goals features may not be comparable across rows.
- No explicit team identity features are included. This avoids sparse categorical overfitting but also limits representational power.
- Final-season statistics must not be added unless they are recomputed fold-by-fold using only past matches.
- Market odds should not be included in the pure NN challenger. A market-informed model should be a separately named experiment.
- Champion model outputs could be useful as stacking/blending meta-features, but only in a separate out-of-fold stacking experiment, not the first pure NN phase.

Recommended feature groups for the next phase:

| Group | Recommendation |
| --- | --- |
| Base rolling form | Keep as a low-complexity sanity input. |
| xG rolling features | Primary pure NN feature group. |
| Full rolling features | Test only after base/xG stages; prior evidence shows richer feature sets can underperform on this sample. |
| Team one-hot | Optional diagnostic only; likely too sparse on one season. |
| Team embeddings | Defer until multi-season data or use only as NN-5 exploratory. |
| Champion probabilities | Separate stacking/blending experiment, not pure NN. |
| Market odds | Separate market-informed benchmark/blend only. |

## Leakage And Validation Risks

Current fold construction is good: training rows are before the test window. Current feature construction is also designed to be sequential for training rows and fitted on the training fold for future fixtures.

The main risks are process risks:

- Repeatedly changing NN hyperparameters based on the same 109-fixture report would overfit the evaluation window.
- If exported data ever includes full-season aggregates, final tables, or post-match stats beyond historical rows, the feature builder will not automatically know that.
- A validation set drawn randomly from a training fold would mix time contexts. Validation should be the latest slice of the training fold or generated by inner expanding-window splits.
- Market odds must stay out of pure model features.

## Benchmarking Standards

The NN challenger should be compared against:

- `champion_dc_xg`.
- Champion-family/statistical leaders such as `dc_fit_rho_each_fold` and `txg_xg_pseudocount_010` when available in that experiment.
- Existing challengers: `regularised_team_strength`, `improved_logistic_regression`, `random_forest`, `poisson_regression`, `logistic_regression`, `elo_baseline`, `naive_outcome_rate`.
- Market-implied benchmark on matched fixtures only, using `evaluation/model_market_comparison.py`.

Primary model-selection metrics:

1. Log loss.
2. Brier score.
3. Accuracy.

Additional NN diagnostics:

- Per-class Brier score for H/D/A.
- Confusion matrix.
- Reliability/calibration by confidence and by class.
- Probability distribution checks: mean probability by class, max-probability histogram, draw-probability distribution, entropy.
- Overconfidence diagnostics: row log-loss tail, count of actual outcome probabilities below 5% and 10%.
- Fold-by-fold metrics, not only aggregate results.
- Train vs validation loss and metric gaps.
- Seed mean/std for promising configurations.

The main question is whether the model improves probability quality, especially on draws, underdogs, and uncertainty. Accuracy-only improvement is not enough.

## Proposed Rigorous Neural Network Phase

### Stage NN-0: Logistic Baseline Sanity Check

- Architecture: no hidden layer, softmax regression over the same scaled feature matrix.
- Purpose: prove the preprocessing, target encoding, folds, metrics, and diagnostic output are correct.
- Keep if it matches or closely tracks `improved_logistic_regression` behaviour.
- Reject the experiment harness if this baseline diverges inexplicably from the existing logistic model.

### Stage NN-1: Tiny MLP

- Architecture: one hidden layer, 8 or 16 units, `tanh` or ReLU if the implementation supports it.
- Regularisation: L2/weight decay required.
- Early stopping: required on a time-aware validation slice.
- Purpose: test whether minimal nonlinearity helps.
- Keep only if log loss and Brier are competitive with the logistic baseline and overconfidence does not worsen.

### Stage NN-2: Moderate MLP

- Architecture: one hidden layer, 32 or 64 units.
- Purpose: capacity test against NN-1.
- Expected risk: high overfitting on current sample.
- Keep only if fold-by-fold and seed-averaged probability metrics improve.

### Stage NN-3: Two-Layer MLP

- Architecture examples: `[32, 16]`, `[64, 32]`.
- Dropout: allowed only after confirming train-validation divergence.
- Purpose: test whether additional representation depth helps.
- Treat as exploratory unless it beats smaller models on log loss across folds and seeds.

### Stage NN-4: Regularised MLP

- Compare L2 values, dropout values, and early-stopping patience.
- Purpose: diagnose whether poor probability quality is regularisation-sensitive.
- Output should explicitly show train/validation divergence and row-level overconfidence.

### Stage NN-5: Optional Embedding/Hybrid Model

- Architecture: team embeddings for home/away teams plus engineered numeric features.
- Strong warning: one-season team embeddings can memorize teams rather than generalise.
- Only justified with multi-season data or as a labelled exploratory experiment.
- Not eligible for promotion on the current one-season evidence alone.

## Controlled Hyperparameter Search

Do not run a full combinatorial grid. Use staged, predeclared search.

Candidate dimensions:

- Hidden layers: `[]`, `[8]`, `[16]`, `[32]`, `[64]`, `[32, 16]`, `[64, 32]`.
- L2/weight decay: `0`, `1e-5`, `1e-4`, `1e-3`.
- Dropout: `0`, `0.1`, `0.2`.
- Learning rate: `1e-3`, `3e-4` for PyTorch; keep separate settings if using the existing NumPy trainer.
- Batch size: `8`, `16`, `32`.
- Early stopping patience: `10`, `20`, `30`.
- Seeds: at least 3 for promising configs.

Practical search design:

1. Run NN-0 and NN-1 on one predeclared feature group, preferably `xg`.
2. Keep only configurations within a small tolerance of the logistic baseline on validation log loss.
3. Expand to NN-2 only if NN-1 is stable across folds.
4. Add dropout only when train loss is good and validation loss is worse.
5. Run 3 seeds for shortlisted configurations.
6. Evaluate final shortlisted configs once on the same outer folds used by champion comparisons.

Selection must prioritise log loss, then Brier, then accuracy.

## Recommended Implementation Route

Use PyTorch only if the project is willing to add a heavier dependency for research experiments. Current `requirements.txt` includes NumPy and pandas but not scikit-learn or PyTorch. The existing logistic model optionally uses scikit-learn when available, but has an internal fallback. The current NN deliberately avoids extra dependencies.

Recommendation for this repository:

- Short term: extend the current NumPy challenger into a rigorous experiment harness. This keeps dependencies light and is enough for NN-0 to NN-2, early stopping, L2, multi-seed runs, and diagnostics.
- Medium term: add PyTorch only if Stage NN-0 to NN-2 reveals credible NN-specific signal or if team embeddings become justified by more seasons/features.
- Avoid scikit-learn `MLPClassifier` as the main route. It is convenient, but weaker for dropout, embeddings, custom diagnostics, and detailed training curves.

This recommendation optimises for scientific rigour rather than tooling ambition.

## Implementation Roadmap

Phase 1: Audit/report only.

- Deliver this document.
- Do not modify `champion_dc_xg` or `model/wsl_xg_model.py`.

Phase 2: Experiment harness.

- Add a separate NN experiment runner, for example `scripts/run_neural_network_experiments.py`.
- Add configurable architecture definitions for `[]`, `[8]`, `[16]`, `[32]`, `[64]`, `[32, 16]`, `[64, 32]`.
- Add time-aware inner validation split or inner expanding-window validation.
- Add early stopping, L2, optional dropout if supported, and multi-seed execution.
- Export per-fold metrics, train/validation history, parameter counts, seed summaries, and row-level predictions.
- Implementation guide: `docs/neural_network_experiments.md`.

Phase 3: Diagnostics.

- Add per-class Brier, confusion matrix, probability distribution summaries, and overconfidence tables.
- Compare against champion, existing challengers, and market benchmark on matched fixtures where odds exist.
- Produce `docs/neural_network_experiments.md` and reports under `reports/`.

Phase 4: Optional extension.

- Only after enough evidence or more data, test PyTorch embeddings/hybrid model.
- Keep this clearly labelled exploratory.

## Go/No-Go Criteria

Go forward with more NN investment only if all are true:

- NN-0 behaves as expected and matches the logistic baseline closely.
- NN-1 or NN-2 improves validation log loss and Brier score over NN-0 across folds.
- Shortlisted configs are stable across at least 3 seeds.
- The final outer-fold comparison improves or nearly matches `improved_logistic_regression` on log loss and Brier without worse overconfidence.
- It shows a specific useful niche, such as better draw probabilities, underdog handling, or calibrated uncertainty.

Do not promote or expand NN complexity if any are true:

- It improves accuracy but worsens log loss or Brier.
- It produces more high-confidence wrong predictions than simpler models.
- Performance depends strongly on a single seed.
- It beats a baseline only after repeated manual tuning on the same final window.
- It cannot beat or explainably complement `improved_logistic_regression`, `random_forest`, or `regularised_team_strength`.

Production promotion should require beating `champion_dc_xg` or a champion-family candidate on probability quality across repeated time-aware evaluations and preferably shadow/live-style runs. The current neural-network evidence does not meet that bar.

## Final Answer To The Central Question

Given the current WSL dataset and feature set, the strongest honest case for a neural network challenger is as a disciplined nonlinear diagnostic over leakage-safe rolling xG/form features. It can test whether small nonlinear interactions help where logistic regression and trees are limited, and it can reveal whether neural models have a niche on draws, underdogs, or uncertainty.

Where it breaks down is data volume and calibration. With roughly one season of evaluated fixtures and small per-team histories, a neural network can easily learn sharp class boundaries that look useful for accuracy but produce poor probabilities. The first proof-of-concept already shows that pattern: acceptable accuracy, worse log loss, and highly confident misses. Until the dataset grows or a controlled NN phase shows stable probability-quality gains, neural networks should remain research-only challengers, not promotion candidates.
