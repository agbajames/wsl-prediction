# Improved Logistic Regression Challenger

`improved_logistic_regression` starts Phase 8B: feature-based machine learning
challengers. It is offline/evaluation-only and does not change the operational
`champion_dc_xg` model.

## Features

The model keeps the original logistic baseline features and adds a conservative
set of leakage-safe football features:

- recent rolling points and goal difference over a four-match window
- season-to-date match counts
- season-to-date xG for and against when xG columns exist
- recent rolling xG difference when xG columns exist
- simple opponent-strength proxies based on points and goal difference allowed

If xG columns are missing, the builder falls back to goals for those xG-derived
features so tests and local evaluation do not require live external services.

## Leakage Prevention

Training features are built sequentially in match-date order. For each training
row, the feature builder computes features before updating team histories with
that match result. Prediction features are built from summaries fitted only on
the supplied training fold.

## Conservative Scope

The feature set is intentionally small. There is no broad hyperparameter search,
no high-capacity model, and no production promotion. The default uses stronger
regularisation than the first logistic challenger.

## Ablation Groups

The same model code supports these predeclared groups:

- `base`
- `form`
- `xg`
- `opponent`
- `full`

The registry entry uses `xg` because the first ablation run showed the full
feature set was less stable on one WSL season. Ablations can instantiate the
model with a different `feature_group`.

## Limitations

The current comparison uses one WSL season and 109 evaluated matches. Any gains
should be treated as directional until confirmed through future shadow/live-style
validation.
