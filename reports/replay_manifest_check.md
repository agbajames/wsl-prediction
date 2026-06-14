# WSL 2025-26 Replay Manifest Check

## Replay Scope

- Weeks intended for replay: 2-22
- Week 1 excluded from this baseline because it requires historical priors or a previous-season baseline.
- Expected replay fixture count: 126 (21 matchweeks x 6 fixtures).
- Monte Carlo simulation is deferred until baseline replay metrics are captured.

## Data-Derived Fixture Windows

| round_label | week | min_match_date | max_match_date | fixture_count | completed_count | notes |
| --- | ---: | --- | --- | ---: | ---: | --- |
| _No live fixture data loaded_ |  |  |  |  |  | Run `python scripts/inspect_matchweek_windows.py --output reports/replay_manifest_check.md` with Supabase credentials. |

## Manifest Verification Findings

- Ready for replay: no
- Manifest weeks missing from data: pending live inspection
- Data rounds missing from manifest: pending live inspection
- Date mismatches: pending live inspection
- Fixture-count mismatches: pending live inspection
- Unverified manifest windows: Matchweeks 2-22 remain placeholders until verified
- Week 1 historical-prior required: true

## Week-By-Week Replay Steps

1. Run `python scripts/inspect_matchweek_windows.py --output reports/replay_manifest_check.md`.
2. Verify Matchweeks 2-22 have six fixtures each and no date mismatches.
3. Update `dashboard/matchweek_manifest.py` only with data-derived or officially verified dates.
4. Start the API with `python -m uvicorn api.main:app --env-file .env`.
5. Start Streamlit with `PREDICTION_API_BASE_URL=http://localhost:8000 API_KEY=$API_KEY python -m streamlit run dashboard/app.py`.
6. In the dashboard, generate predictions for Matchweeks 2 through 22 in order.
7. Confirm each run appears in prediction history and therefore in `prediction_runs`.
8. Run evaluation after predictions are logged, using the dashboard-generated command or `python -m evaluation.run_evaluation --start-date <week_predict_from> --persist --run-trigger dashboard-season-2025-26-week-XX-evaluation`.
