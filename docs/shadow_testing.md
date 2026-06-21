# Shadow Prediction Artefacts

Phase 8E-1 adds an evaluation-only framework for saving pre-match predictions before WSL fixtures are played and replaying those saved rows after results are known. It does not promote a model, change production prediction behaviour, or modify `model/wsl_xg_model.py`; `champion_dc_xg` remains the operational/reference model.

Phase 8E-2 turns that framework into the first real-run workflow. If real upcoming WSL fixture data is not available locally, stop after validating the template or prepared fixture file. Do not fabricate fixtures, predictions or results.

Phase 8E-3 adds a local fixture normalisation helper for converting an approved local upcoming-fixtures CSV into the canonical shadow fixture schema. It does not fetch, scrape or create fixtures.

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

See `data/templates/shadow_fixtures_template.csv` for the real-run input shape and `data/samples/shadow_fixtures_sample.csv` for a safe synthetic/sample input. The template is not a prediction artefact and must have every placeholder replaced before generation.

## Normalise Raw Fixture CSVs

Use `scripts/normalise_shadow_fixtures.py` when the available source file uses different column names from the shadow template. The script reads a local CSV only; it never fetches fixtures from the web or Supabase.

Accepted required column-name variants:

| canonical field | accepted raw names |
| --- | --- |
| `fixture_date` | `fixture_date`, `match_date`, `date` |
| `home_team` | `home_team`, `home`, `home_team_name` |
| `away_team` | `away_team`, `away`, `away_team_name` |

Accepted optional raw names include `fixture_id`, `id`, `match_id`, `game_id`, `round_label`, `round`, `matchweek`, `match_week`, `season`, `competition`, `competition_name`, `venue`, `stadium`, `kickoff_time`, `kickoff`, `time`, `source_notes` and `notes`.

The normaliser validates that fixture dates are parseable, home and away teams are present, teams are not identical, and duplicate fixture rows or duplicate fixture IDs are flagged. Completed-result columns such as `home_goals`, `away_goals` or `actual_outcome` are not required for upcoming fixtures and are not copied into the normalised output.

Example with the safe sample input:

```powershell
.\.venv\Scripts\python.exe scripts\normalise_shadow_fixtures.py `
  --input data\samples\raw_shadow_fixtures_sample.csv `
  --output data\shadow\upcoming_wsl_fixtures.csv
```

For a real run, replace the sample input with an approved local raw fixture CSV. Keep raw private exports and unapproved real fixture files out of Git.

## First Real Shadow Runbook

1. Prepare a fixture input file from a real, approved upcoming WSL fixture source.
   - Save local working inputs outside Git unless they are public and approved for commit.
   - Use `data/templates/shadow_fixtures_template.csv` as the column guide.
   - Include one row per upcoming fixture and prefer stable source `fixture_id` values over fallback IDs.
2. If the source column names differ from the template, normalise the raw CSV.
   - Run `scripts\normalise_shadow_fixtures.py --input path\to\raw_upcoming_fixtures.csv --output data\shadow\upcoming_wsl_fixtures.csv`.
   - Review the output before any prediction generation.
3. Confirm the historical export contains only matches known before each fixture date.
   - The usual local path is `data/exports/wsl_match_data.csv`.
   - The generation code trains each model on `match_date < fixture_date` for every fixture group.
4. Choose the candidate set.
   - Start with the core tracking set below unless a candidate is temporarily broken.
   - Omit unavailable candidates with explicit repeated `--model` arguments; do not change production prediction code to make a shadow run pass.
5. Validate fixtures before generating predictions.
   - This confirms the fixture schema and prints the planned model set and current git SHA.
6. Generate a timestamped artefact only from real upcoming fixture input.
   - Write artefacts under `reports/shadow_predictions/`.
   - Use UTC-style filenames, for example `shadow_predictions_20260905T103000Z.json`.
7. Verify provenance immediately after generation.
   - Open the JSON metadata and confirm `git_sha`, `generated_at`, `history_csv`, `fixtures`, `models` and `min_train_matches`.
   - Confirm every prediction row has a non-empty `prediction_timestamp`, `git_sha`, `model_config` and stable fixture key.
8. Replay after results are known.
   - Prepare a results file with matching `fixture_id` values and either `actual_outcome` or `home_goals`/`away_goals`.
   - Write replay output under `reports/shadow_predictions/`.
   - Treat pending fixtures as unevaluated; only completed rows contribute to metrics.

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

## Candidate Model Selection

The core shadow tracking set is:

| candidate | status in current framework | note |
| --- | --- | --- |
| `champion_dc_xg` | callable | Operational/reference model; remains the first tracked row and is not promoted or modified by shadow testing. |
| `dc_fit_rho_each_fold` | callable | Best probability-quality candidate overall from Phase 8A. |
| `txg_xg_pseudocount_010` | callable | Most balanced champion-family candidate from Phase 8A. |
| `blend_dc_fit_txg_50_50` | callable | Best non-market blend so far; evaluation-only fixed blend of the two champion-family variants. |
| `regularised_team_strength` | callable | Strongest standalone statistical challenger. |
| `improved_logistic_regression` | callable | Best feature-based ML challenger on Brier/log loss. |
| `random_forest` | callable, optional | Useful tree-based challenger, but not part of any production promotion decision. |

The neural-network proof of concept remains research-only and is intentionally not part of the default shadow run. If a future checkout cannot call one of the listed candidates through `evaluation.shadow`, document the omission in the run notes and pass only the callable candidates with repeated `--model` arguments.

## Generate

Validate the fixture file first. This command does not fit models and does not write an artefact:

```powershell
.\.venv\Scripts\python.exe scripts\generate_shadow_predictions.py `
  --fixtures data\shadow\upcoming_wsl_fixtures.csv `
  --validate-fixtures-only
```

For a real run, replace the template path with the approved real upcoming fixture file:

```powershell
.\.venv\Scripts\python.exe scripts\generate_shadow_predictions.py `
  --history-csv data\exports\wsl_match_data.csv `
  --fixtures data\exports\upcoming_wsl_fixtures.csv `
  --output reports\shadow_predictions\shadow_predictions_YYYYMMDDTHHMMSSZ.json
```

The generation script fits each selected candidate only on historical matches before each fixture date. If a Phase 8 candidate is not callable through the offline evaluation providers in a future checkout, omit it with explicit `--model` arguments rather than changing production prediction code.

To run a smaller explicit set:

```powershell
.\.venv\Scripts\python.exe scripts\generate_shadow_predictions.py `
  --history-csv data\exports\wsl_match_data.csv `
  --fixtures data\exports\upcoming_wsl_fixtures.csv `
  --model champion_dc_xg `
  --model dc_fit_rho_each_fold `
  --model txg_xg_pseudocount_010 `
  --output reports\shadow_predictions\shadow_predictions_YYYYMMDDTHHMMSSZ.json
```

## Replay

When results are available, prepare a results CSV or JSON with `fixture_id` plus either `actual_outcome` or `home_goals` and `away_goals`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_shadow_predictions.py `
  --predictions reports\shadow_predictions\shadow_predictions_YYYYMMDDTHHMMSSZ.json `
  --results data\exports\wsl_results.csv `
  --output-json reports\shadow_predictions\shadow_replay_YYYYMMDD.json
```

Pending fixtures remain in the replay payload with no metrics impact. Completed fixtures are ranked with Brier score, log loss and accuracy using the existing evaluation comparison helpers.

## Safety Checklist

- `model/wsl_xg_model.py` is unchanged.
- `champion_dc_xg` remains the operational/reference model.
- No model is promoted from a shadow run alone.
- Fixture templates and samples are clearly labelled and cannot be mistaken for real predictions.
- Raw private fixture exports and normalised real fixture files stay out of Git unless explicitly approved.
- Real prediction artefacts are generated only from real upcoming fixture input.
- `.env`, credentials, Supabase keys and private exports stay out of Git.
