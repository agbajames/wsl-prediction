# Logistic Regression Challenger

`logistic_regression` is the first machine-learning challenger model in the
evaluation roadmap. It is intentionally simple, transparent, and cheap to run
before adding more complex tree models, neural networks, market blending, or
external-league transfer.

## Features

The model uses team-form features built from historical match results:

- home and away points per match
- home and away goals for per match
- home and away goals against per match
- points-per-match and goal-difference-rate differences
- a home indicator

Optional xG columns are not required in this first challenger. Missing optional
features therefore do not block local mocked DataFrame tests.

## Leakage Prevention

Training features are built sequentially in match-date order. For each training
row, the feature builder uses only matches already processed before that row.
Prediction features are then built from summaries fitted on the training fold
only. The rolling backtest framework still controls the date split, so test
window matches are not included in feature construction.

## Comparison Role

The logistic challenger implements the same model protocol as the champion,
naive baseline, and Elo baseline. It can be evaluated on identical rolling
folds and reported with the same Brier score, log loss, accuracy, calibration,
and failure-analysis artefacts.

## Limitations

One-season WSL samples are small, and multinomial logistic regression can be
unstable when a fold has few matches or too few outcome classes. In those cases
this implementation falls back to the naive outcome-rate baseline. The model is
a first ML reference point, not a production replacement for the frozen
champion.

