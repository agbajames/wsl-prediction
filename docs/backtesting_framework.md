# Backtesting Framework

Rolling, time-based backtesting is needed because football prediction is a
forecasting problem. A model should only learn from matches that would have
been known before the fixtures it is predicting.

Random train/test splits are unsafe here. They can put later matches into the
training set while earlier matches are in the test set, which leaks future team
strength, injuries, form, schedule effects, and rescheduled fixture context into
the evaluation.

## Folds

A fold is one leakage-safe train/test split:

- train rows are matches strictly before the test window
- test rows are matches inside the test window
- optional round-label filters can narrow the test rows when labels are present
- fold metadata records dates, sizes, and preserved round labels

The fold builder is deterministic for the same input data and configuration.
The same fold list can be reused for the champion model and future challenger
models, so comparisons happen on identical fixtures.

## Leakage Prevention

`evaluation/backtesting.py` uses date-based windows. For a test window starting
on a given date, training data must have `match_date < test_start`. Matches on
the test date or inside the prediction window are never included in that fold's
training data.

This matters for the WSL because postponed and rearranged fixtures can carry a
round label from one part of the season while being played later. Date remains
the primary split key; round labels are preserved as metadata and may be used as
an optional test filter, but they do not make future matches eligible for
training.

## Champion and Challengers

The framework works with models that implement the lightweight protocol in
`models/base.py`. The current champion adapter, `champion_dc_xg`, can be run
through the same folds that future challengers will use. That gives evaluation
metrics a stable baseline instead of allowing each model to choose its own
train/test split.

## Current Limitations

The WSL sample is small, so some folds may have too little history and can be
skipped by configuration. Wide test windows give more fixtures per fold but
fewer independent evaluation points. Narrow windows are more realistic for
weekly forecasting but can be noisy.

Rescheduled fixtures are handled by match date, which is safest for leakage,
but it means round labels may not align with chronological windows. That is
expected and should be documented in experiment outputs.

