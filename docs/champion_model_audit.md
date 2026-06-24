# WSL Prediction Engine - Champion Model Audit

## 1. Executive Summary

The current reference model is `champion_dc_xg`, an xG-driven Dixon-Coles/Poisson model exposed through `models/champion_dc_xg.py` and implemented in `model/wsl_xg_model.py`. It estimates team attacking and defensive strength from non-penalty xG, applies a low-score Dixon-Coles adjustment, models penalty xG separately, and outputs home-win, draw and away-win probabilities.

Across the main 109-fixture local evaluation window, the champion remains the strongest operational model in the repository: `reports/model_comparison_first_run.md` gives approximately 62.4% accuracy, 0.505 Brier score and 0.867 log loss. The matched-fixture market-implied benchmark in `reports/model_vs_market_comparison_2025_26.md` is slightly ahead at approximately 63.3% accuracy, 0.489 Brier score and 0.837 log loss. The broader market-only report over 132 fixtures is stronger again, with approximately 65.2% accuracy, 0.471 Brier score and 0.813 log loss in `reports/market_benchmark_2025_26.md`.

It is reasonable to keep `champion_dc_xg` as the current reference model. It beats the tested simple baselines and feature-based ML challengers, is interpretable, and is close to the external market probability reference despite not using market odds as training features or production inputs. The key audit message is that the champion model is a credible and useful reference, but the remaining gap is mostly about probability calibration, richer context, and robustness across more WSL seasons or shadow/live-style evaluations.

## 2. Model Overview

The champion adapter in `models/champion_dc_xg.py` is deliberately thin. `ChampionDCXGModel.fit()` calls `estimate_team_strengths()`, `estimate_penalty_rates()` and `_resolve_rho()`. `ChampionDCXGModel.predict()` calls `predict_fixtures()` and adds common model identity metadata. This keeps the production model logic in one place while making the model usable by the shared evaluation framework.

The underlying model in `model/wsl_xg_model.py` uses match-level WSL data with dates, teams, goals, total xG and non-penalty xG. `split_played_future()` uses date-based training and prediction windows, not round labels, which is important for postponed and rearranged fixtures.

Team strength is estimated by `estimate_team_strengths()` through weighted ridge regression on `log(np_xG + xg_pseudocount)`. The design matrix includes an intercept, home advantage, home/away attacking effects and opponent defensive effects. Recent matches are upweighted by `compute_time_weights()` using the configured half-life, currently 60 days by default. Attack and defence parameters are centred after fitting for interpretability.

The model uses non-penalty xG as the main quality signal. Penalty xG is handled separately in `estimate_penalty_rates()`, which derives penalty xG as total xG minus non-penalty xG and shrinks team penalty rates towards league averages. This is a sensible small-sample choice because penalties are rare and can distort team-strength estimates if mixed directly into open-play quality.

For a fixture, `TeamStrengths.expected_goals()` produces open-play expected goals for the home and away team. `compute_scoreline_matrix()` builds a Poisson scoreline matrix, applies `dixon_coles_adjustment()` to low-scoring open-play outcomes, and then convolves independent penalty components onto the scoreline distribution. `wdl_from_matrix()` converts the scoreline matrix into home-win, draw and away-win probabilities.

Home advantage is represented as a fitted log-scale coefficient in the team-strength regression. Unknown teams default to zero-centred attack/defence effects, which is operationally convenient but means promoted-team handling is mostly implicit rather than a dedicated prior.

The default configuration is held in `ModelConfig`: ridge strength `alpha=0.15`, 60-day time decay, fixed `rho=-0.13`, score grid up to 8 goals, moderate penalty-rate shrinkage and a small xG pseudocount. `ChampionDCXGModel.from_config()` can also load serialised configuration and optionally fit `rho` when configured.

## 3. What Is Working Well

The model is strong because it encodes football structure directly. It does not ask a generic classifier to infer score generation from a small sample; it estimates team attacking and defensive quality, applies home advantage, generates scoreline probabilities, and then aggregates those scorelines into 1X2 probabilities.

Using xG is a major advantage over raw goals alone. Goals are sparse and noisy in a 22-match WSL league season, while non-penalty xG carries more information about chance quality and repeatable team performance. The model still evaluates against real goals, but its strength estimates are less hostage to finishing variance.

Regularisation and time decay are also doing useful work. Ridge shrinkage helps avoid extreme team parameters from small samples, while time decay lets the model respond when team strength changes during a season. The penalty component is appropriately shrunk because penalties are too rare for stable team-specific rates without pooling.

The model is interpretable. It can explain predictions through team attack, team defence, home advantage, expected goals, draw probability and top scorelines. That makes it more useful as a reference model than a black-box challenger with similar headline accuracy.

It is competitive without using market odds as input features. The market comparison code in `evaluation/market_benchmark.py` and `evaluation/model_market_comparison.py` explicitly treats odds as an external evaluation-only probability reference. The champion therefore appears to be learning genuine football signal from match/xG data rather than rediscovering market information from odds.

## 4. Evaluation Summary

The main model comparison in `reports/model_comparison_first_run.md` evaluates 109 matches:

| model | Brier score | log loss | accuracy |
| --- | ---: | ---: | ---: |
| `champion_dc_xg` | 0.5052 | 0.8669 | 0.6239 |
| `logistic_regression` | 0.5562 | 0.9560 | 0.5780 |
| `elo_baseline` | 0.5676 | 0.9596 | 0.5596 |
| `naive_outcome_rate` | 0.6401 | 1.0576 | 0.4587 |

The matched model-vs-market comparison in `reports/model_vs_market_comparison_2025_26.md` uses 109 matched fixtures:

| reference | Brier score | log loss | accuracy |
| --- | ---: | ---: | ---: |
| `market_implied_benchmark` | 0.4885 | 0.8369 | 0.6330 |
| `champion_dc_xg` | 0.5052 | 0.8669 | 0.6239 |

The broader market-only benchmark in `reports/market_benchmark_2025_26.md` covers 132 market rows after excluding one non-league row and reports Brier 0.4712, log loss 0.8127 and accuracy 0.6515. This is not directly identical to the 109-fixture matched comparison because the fixture set differs.

Later Phase 8A/8B reports show that some champion-family variants are close or slightly better on the same window. `reports/statistical_challengers_phase_8a_summary.md` records `dc_fit_rho_each_fold` at Brier 0.5027 and log loss 0.8623, while `txg_xg_pseudocount_010` reaches accuracy 0.6330. `reports/feature_ml_challengers_phase_8b_summary.md` shows improved logistic regression and random forest still trailing the champion-family/statistical leaders. The repository has therefore already found promising candidates, but has not promoted them.

Important caveats:

- The main evaluation covers one WSL season window and 109 fixtures, so small metric differences may be unstable.
- The market-only and matched-market reports use different fixture counts, so their headline values should not be blended casually.
- Fixture matching depends on date and normalised team names in `evaluation/model_market_comparison.py`.
- Accuracy is useful but coarse; Brier score, log loss, calibration and row-level failure analysis are more informative for probability forecasts.
- Market probabilities are an evaluation-only reference and are not used as model training features.

## 5. Where the Market Is Still Better

The market-implied benchmark is still ahead, mainly on probability quality. Its advantage in Brier score and log loss suggests better calibration of how confident to be, not simply better top-class selection.

The market has access to broader and later information than the current model: injuries, suspensions, line-ups, transfers, manager changes, squad rotation, fixture congestion, Champions League context, motivation, venue changes, weather and late team news. It also benefits from collective intelligence and price discovery, where multiple informed participants move prices towards a consensus.

The champion model only sees structured match/xG history. That is enough to be competitive, but it cannot know that a key striker is absent, that a team has just changed manager, or that a European fixture is likely to alter team selection unless those effects have already appeared in past xG.

There may also be calibration issues. The comparison report shows the champion assigning higher average draw probability than the market and making some draw picks, while the market makes none in the matched report. The model may still be too confident or not confident enough in specific scoreline regimes, especially around dominant favourites, away favourites and draws.

## 6. Possible Edge Areas for Future Improvement

### 6.1 Feature and Data Improvements

- Add rolling attacking and defensive xG trends over predeclared windows, tested for leakage.
- Add recent form based on xG differential rather than only results.
- Track squad availability, key player absences, keeper availability and striker availability.
- Model transfer-window squad strength changes explicitly, especially for promoted teams and top clubs.
- Add rest days, travel, fixture congestion and Champions League proximity.
- Split home and away team-strength signals where sample size permits.
- Add promoted/relegated team priors instead of relying only on zero-centred unknown-team defaults.
- Add managerial-change flags and post-change recency features.
- Add strength-of-schedule adjustments for recent xG form.
- Track finishing and goalkeeping over/under-performance as separate cautionary signals rather than letting them fully override xG.

### 6.2 Model Improvements

- Continue testing time-decay variants, but require shadow validation because one-window differences are small.
- Carry forward `dc_fit_rho_each_fold` and `txg_xg_pseudocount_010` as champion-family candidates, as identified in Phase 8A.
- Introduce dynamic team strengths or state-space updates if more historical data becomes available.
- Add Bayesian shrinkage for team attack/defence and promoted-team priors.
- Test explicit draw calibration, but note that the existing calibration and draw-adjustment reports did not justify promotion.
- Test non-market ensembles carefully. `reports/non_market_blending_first_run.md` shows `blend_dc_fit_txg_50_50` was close, but not enough for promotion.
- Test post-model calibration only through proper out-of-fold evaluation; `reports/champion_calibration_first_run.md` shows the first temperature-plus-shrinkage attempt worsened trial Brier and log loss.

### 6.3 Evaluation Improvements

- Add reliability diagrams and class-specific calibration, not only max-confidence bins.
- Report per-class accuracy and per-class log loss for home, draw and away outcomes.
- Add top-four versus rest, promoted-team, and favourite/underdog subgroup evaluation.
- Analyse high-confidence misses separately from ordinary misses.
- Build probability bucket analysis for each outcome class.
- Run rolling matchweek-by-matchweek reports to identify drift.
- Compare champion, variants, simple baselines and market-implied benchmark over multiple seasons once data exists.
- Add matched-fixture market gap analysis showing where the model assigns more probability to the realised outcome than the external market probability reference.

### 6.4 WSL-Specific Edges

The WSL has a small league, large gaps between top and lower-table clubs, uneven squad depth, high transfer impact and limited public data compared with men's football. That makes xG-based structure valuable, but also means a small number of injuries or transfers can materially change team strength.

Top clubs may be affected by Champions League congestion and rotation. Promoted teams are uncertain because prior WSL xG history is sparse or absent. Public odds markets for women's football may be less liquid than major men's competitions, so a well-maintained model may still find local edges, but those claims should be tested against timestamped pre-match market snapshots rather than assumed.

## 7. Risk Areas and Current Limitations

- The main comparison sample is small: approximately 109 evaluated WSL fixtures.
- Results may be overfit to one season or one evaluation window.
- Player-level information is not currently part of the champion model.
- Draw probability may need more targeted diagnostics and calibration.
- Data quality matters heavily: inconsistent xG sources or missing non-penalty xG would directly affect strength estimates.
- Structural changes between seasons, promotions and squad rebuilds are difficult for a mostly team-history model.
- Unknown or newly promoted teams default towards average strengths unless handled elsewhere.
- Market comparisons depend on odds source quality, no-vig conversion, timing and licensing.
- Some roadmap documentation has been updated by later experiments, so older statements about "next" phases should be read alongside the current Phase 8A/8B reports.

## 8. Recommended Next Steps

### Immediate - Low Risk

| recommendation | why it may help | metric to improve | how to test | risk |
| --- | --- | --- | --- | --- |
| Produce class-specific calibration and reliability diagrams | Identifies whether errors are concentrated in home, draw or away probabilities | Brier, log loss, calibration gap | Read-only analysis from existing prediction rows | Low |
| Run top-four/rest and promoted-team subgroup evaluation | Tests WSL-specific failure modes | Subgroup log loss and actual-outcome probability | Slice existing row-level results by team groups | Low |
| Analyse matched market gaps by fixture type | Finds where the model is systematically below the external market probability reference | Log loss delta, actual probability delta | Use `evaluation/model_market_comparison.py` outputs | Low |
| Audit high-confidence misses | Separates bad luck from overconfident model structure | High-confidence log loss | Review worst misses in existing reports | Low |

### Medium-Term - Moderate Engineering

| recommendation | why it may help | metric to improve | how to test | risk |
| --- | --- | --- | --- | --- |
| Shadow-test `dc_fit_rho_each_fold` and `txg_xg_pseudocount_010` | They slightly improved first-run metrics but need out-of-sample confirmation | Brier and log loss | Generate timestamped predictions before fixtures, then replay results | Medium |
| Add rolling xG form and strength-of-schedule features | Captures team changes faster than season-level strengths | Brier, log loss, subgroup accuracy | Predeclare windows and compare against champion in walk-forward evaluation | Medium |
| Add fixture congestion and rest features | WSL top clubs may rotate around European fixtures | Log loss on top-club fixtures | Backtest with leakage-safe fixture-calendar features | Medium |
| Add promoted-team priors | Reduces uncertainty for teams with little WSL history | Early-season log loss | Compare early-season/promoted-team subsets | Medium |
| Test fixed non-market blends from champion-family candidates | Blends can smooth small model-specific errors | Brier and log loss | Fold-safe blend evaluation, then shadow testing | Medium |

### Longer-Term - Research/Advanced

| recommendation | why it may help | metric to improve | how to test | risk |
| --- | --- | --- | --- | --- |
| Build dynamic/Bayesian team-strength model | Better handles small samples and changing team strength | Brier, log loss, calibration | Multi-season backtest with priors and posterior updates | High |
| Add player-level availability model | Captures information the market likely prices quickly | Log loss, upset/favourite failure analysis | Requires reliable squad/line-up data and timestamping | High |
| Use multi-season WSL or external league adaptation | Reduces one-season overfitting and improves priors | Stability across seasons | Rolling season-by-season validation | High |
| Market-aware research layer | Tests whether the model adds signal beyond the market-implied benchmark | Log loss delta versus market | Only with licensed, timestamped pre-match odds snapshots | High |

## 9. Suggested Experiments

| experiment | hypothesis | implementation idea | expected benefit | evaluation method | success criteria |
| --- | --- | --- | --- | --- | --- |
| Time-decay champion variant | A different recency half-life better captures WSL team changes | Re-run a small predeclared grid such as 45/60/90 days and carry forward existing Phase 8A variants | Better current-strength estimates | Walk-forward plus shadow validation | Improves Brier/log loss without worsening calibration |
| Calibrated champion model | Probability scaling can improve calibration without changing picks | Fit calibration only on earlier folds and test on later folds | Lower log loss | Out-of-fold calibration experiment | Beats uncalibrated champion on Brier and log loss |
| Champion plus market gap analysis | The model may be stronger in specific fixture types | Analyse rows where model actual-outcome probability exceeds market reference | Identifies possible local edge categories | Matched market comparison slices | Repeatable positive log-loss delta in a predeclared subgroup |
| Top-four versus rest subgroup evaluation | WSL hierarchy may create different error regimes | Tag fixtures by club tier and compare metrics | Better diagnosis of favourite and underdog handling | Existing row-level evaluation | Clear subgroup where model underperforms and can be targeted |
| Draw-specific calibration | Draw probabilities may be misaligned by fixture type | Analyse draw probability buckets and fixed draw transformations | Better draw log loss | Fold-safe draw calibration and reliability diagrams | Improves draw Brier/log loss without damaging overall metrics |
| Fixture congestion feature test | Rest and European congestion affect top-club selection | Add rest-day and congestion features to challenger models | Better top-club predictions | Walk-forward backtest | Improves top-club subgroup log loss |
| Rolling xG form feature test | Recent xG trend adds signal beyond static strength | Add rolling xG for/against and xG differential windows | Faster response to form changes | Predeclared ablation | Improves log loss and avoids overfitting in ablation |
| Ensemble with challenger model | Blending reduces variance and idiosyncratic misses | Fixed or fold-learned blend of champion-family and regularised/statistical challengers | Smoother probabilities | Out-of-fold blend evaluation, then shadow testing | Beats champion on Brier/log loss over multiple windows |
| Promoted-team prior experiment | Promoted-team uncertainty needs explicit shrinkage | Add priors based on promoted status or external rating | Better early-season probabilities | Early-season and promoted-team subgroup evaluation | Lower early-season/promoted-team log loss |

## 10. Final Recommendation

Keep `champion_dc_xg` as the reference/champion model for now. It has the best operational evidence among the currently deployed candidates, beats the simple baselines and feature-based ML challengers, and remains close to the market-implied benchmark on the matched 109-fixture comparison.

The most sensible next path is not a rewrite. It is a measured validation path: deepen calibration and subgroup diagnostics, shadow-test the two best champion-family variants (`dc_fit_rho_each_fold` and `txg_xg_pseudocount_010`), and add carefully timestamped WSL-specific contextual features such as rolling xG form, squad availability, promoted-team priors and fixture congestion. Promotion should require repeatable improvement in Brier score and log loss, not just a small one-window accuracy gain.
