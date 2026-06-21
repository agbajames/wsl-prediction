# Baseline Models

Baseline models are simple challengers used to sanity-check the frozen
champion model. They make it harder to mistake complexity for quality: a new
model should beat transparent reference points before it earns trust.

## Naive Outcome-Rate Baseline

`naive_outcome_rate` learns the historical home win, draw, and away win rates
from the training fold. It predicts those same three probabilities for every
fixture in the test window.

This measures whether a model is doing better than the base rate of WSL match
outcomes. It is deliberately context-free: no team strength, fixture date,
home advantage beyond the historical home-rate, or xG signal is used.

## Elo Baseline

`elo_baseline` maintains simple team ratings from historical match results. It
updates ratings in date order, applies a configurable home-advantage rating
boost, and converts the rating gap into home/draw/away probabilities.

This measures whether a lightweight result-based team-strength model can
compete with the xG-driven champion.

## Why These Come Before ML Challengers

Naive and Elo baselines are interpretable and cheap to evaluate. They help
catch leakage, broken folds, probability-shape issues, and overfitted future
challengers before adding logistic regression, tree models, neural networks, or
market blending.

## Limitations

One-season WSL samples are small. The naive baseline can be noisy early in the
season, and Elo has limited evidence for promoted, new, or heavily changed
teams. Rescheduled fixtures are handled by the rolling backtest dates, but the
baselines themselves only know the training rows they are given.

## Comparison With Champion

Both baselines implement the evaluation model protocol from `models/base.py`.
The rolling backtest framework can fit them on the same folds as
`champion_dc_xg`, and the reporting framework can compare Brier score, log
loss, accuracy, calibration, and failure-analysis views across all models.

