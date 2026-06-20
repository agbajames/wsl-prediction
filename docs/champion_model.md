# Champion Model

The champion model is the current production/default WSL prediction model:
`champion_dc_xg` version `v1`, in the `xg_dixon_coles_poisson` family.

The underlying implementation lives in `model/wsl_xg_model.py`. It estimates
team attack and defence strengths from non-penalty xG, applies the existing
Dixon-Coles low-score adjustment, models penalty xG separately, and generates
fixture-level win/draw/loss probabilities.

## Adapter

`models/champion_dc_xg.py` exposes the existing implementation through a small
evaluation interface:

- model identity: name, family, and version
- frozen/default configuration metadata
- `fit(...)` for played matches
- `predict(...)` for fixture-level predictions
- `fit_from_dataset(...)` for the existing date-based split flow

The adapter calls the existing functions from `model/wsl_xg_model.py`,
including `ModelConfig`, `split_played_future`, `estimate_team_strengths`,
`estimate_penalty_rates`, `fit_rho`, and `predict_fixtures`.

## Non-Goals

The adapter does not change model behaviour, API behaviour, dashboard
behaviour, or prediction outputs. It does not rewrite or duplicate the
champion model logic. It is a compatibility layer for evaluation workflows.

## Frozen Config

The frozen champion config is stored at
`experiments/configs/champion_dc_xg.yaml`. It records the current default
settings, including alpha, decay, rho behaviour, max goals, bootstrap default,
input schema reference, and model identity.

Freezing this champion before adding challenger models gives future evaluation
runs a stable baseline. Challenger models can be compared against the same
identity, configuration, input contract, and output shape rather than against a
moving target.

## Future Comparisons

Future challengers should implement the same lightweight interface from
`models/base.py`. Evaluation runners can then fit each model on the same played
data, predict the same fixtures, and compare metrics such as log loss, Brier
score, calibration, accuracy, and per-match probability quality against
`champion_dc_xg` version `v1`.

