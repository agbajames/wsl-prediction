# Shadow Prediction Artefacts

Phase 8E-1 adds an evaluation-only framework for saving pre-match predictions before WSL fixtures are played and replaying those saved rows after results are known. It does not promote a model, change production prediction behaviour, or modify `model/wsl_xg_model.py`; `champion_dc_xg` remains the operational/reference model.

## Fixture Input Schema

Upcoming fixture files may be CSV or JSON. JSON may be either a list of objects or an object with a `fixtures` list.

Required fields:

| field | description |
| --- | --- |
| `fixture_date` or `match_date` | Scheduled fixture date. Training uses only historical rows with `match_date < fixture_date`. |
| `home_team` | Home team display name. |
| `away_team` | Away team display name. |

Recommended fields:

| field | description |
| --- | --- |
| `fixture_id` | Stable source fixture identifier. If absent, a deterministic date/team fallback ID is created. |
| `round_label` | Competition round or matchweek label. |
| `season` | Season label such as `2026-27`. |

See `data/samples/shadow_fixtures_sample.csv` for a safe sample input.

## Prediction Artefact Schema

Every saved prediction row must include:

| field | description |
| --- | --- |
| `prediction_id` | Deterministic row key combining model, fixture and prediction timestamp. |
| `prediction_timestamp` | UTC timestamp proving when the prediction was generated. |
| `git_sha` | Code revision used for the run. |
| `model_name` | Candidate model or fixed blend name. |
| `model_family` | Evaluation model family. |
| `model_version` | Model/config version. |
| `model_config` | JSON-serialized model configuration. |
| `fixture_id` | Stable or fallback fixture identifier. |
| `fixture_date` | Scheduled fixture date. |
| `home_team`, `away_team` | Teams. |
| `p_home_win`, `p_draw`, `p_away_win` | H/D/A probabilities in unit scale, summing to approximately 1. |
| `predicted_outcome` | Highest-probability H/D/A label. |

The default shadow tracking set is `champion_dc_xg`, `dc_fit_rho_each_fold`, `txg_xg_pseudocount_010`, `blend_dc_fit_txg_50_50`, `regularised_team_strength`, `improved_logistic_regression` and `random_forest`. These are evaluation-only candidates; no production registry or champion promotion is implied.

## Generate

```powershell
.\.venv\Scripts\python.exe scripts\generate_shadow_predictions.py `
  --history-csv data\exports\wsl_matches.csv `
  --fixtures data\samples\shadow_fixtures_sample.csv `
  --output reports\shadow_predictions\shadow_predictions_YYYYMMDDTHHMMSSZ.json
```

The generation script fits each selected candidate only on historical matches before each fixture date. If a Phase 8 candidate is not callable through the offline evaluation providers in a future checkout, omit it with explicit `--model` arguments rather than changing production prediction code.

## Replay

When results are available, prepare a results CSV or JSON with `fixture_id` plus either `actual_outcome` or `home_goals` and `away_goals`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_shadow_predictions.py `
  --predictions reports\shadow_predictions\shadow_predictions_YYYYMMDDTHHMMSSZ.json `
  --results data\exports\wsl_results.csv `
  --output-json reports\shadow_predictions\shadow_replay_YYYYMMDD.json
```

Pending fixtures remain in the replay payload with no metrics impact. Completed fixtures are ranked with Brier score, log loss and accuracy using the existing evaluation comparison helpers.
