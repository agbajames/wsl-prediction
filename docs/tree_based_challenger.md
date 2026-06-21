# Tree-Based Challenger

`random_forest` is the Phase 8B-2 offline tree-based challenger. It tests
whether simple nonlinear interactions among leakage-safe tabular football
features improve on logistic regression.

## Model

The challenger uses a small deterministic random forest implemented with NumPy:

- 50 trees
- maximum depth 3
- minimum 5 samples per leaf
- square-root feature subsampling
- bootstrap sampling with a fixed random seed
- smoothed class probabilities at leaves

No production prediction behaviour changes, and `model/wsl_xg_model.py` remains
untouched.

## Features

The model reuses the `xg` feature group from `improved_logistic_regression`,
because that group was the strongest Phase 8B-1 ablation. Features are generated
sequentially inside each training fold, so a match can only see matches played
before it.

## Why Conservative

The WSL sample is small. A deep or heavily tuned tree model could fit noise in
one season, so this first tree challenger uses fixed shallow defaults and no
hyperparameter search.

## Interpreting Results

This model is evaluation-only. If it improves one metric but worsens another,
that should be treated as a hypothesis for later shadow testing rather than a
production decision.
