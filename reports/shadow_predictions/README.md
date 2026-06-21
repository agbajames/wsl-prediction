# Shadow Prediction Artefacts

This directory is reserved for safe, generated shadow prediction artefacts and replay outputs.

Do not commit files containing credentials, private source data, Supabase keys or non-public fixture feeds. Generated examples are safe to commit only when they contain public or synthetic/sample fixture information and no secrets.

## Commit Rules

- Commit `README.md` and clearly labelled sample/test artefacts only.
- Do not commit raw private exports, `.env` files, Supabase payloads, credentials, or paid/non-public fixture feeds.
- Real shadow prediction artefacts may be committed only when the input fixture list is public or otherwise approved for the repository and the artefact is timestamped with `prediction_timestamp` and `git_sha`.
- Use filenames such as `shadow_predictions_20260905T103000Z.json` and replay filenames such as `shadow_replay_20260915.json`.
- Keep generated local experiments out of Git until their provenance is checked.
