# WSL 2025-26 Replay Manifest Check

## Replay Scope

- Weeks intended for replay: 2-22
- Week 1 excluded from this baseline because it requires historical priors or a previous-season baseline.
- Expected replay fixture count: 126 (21 matchweeks x 6 fixtures).
- Monte Carlo simulation is deferred until baseline replay metrics are captured.

## Data-Derived Fixture Windows

| round_label | week | min_match_date | max_match_date | fixture_count | completed_count | notes |
| --- | ---: | --- | --- | ---: | ---: | --- |
| R1 | 1 | 2025-09-05 | 2025-09-07 | 6 | 6 |  |
| R2 | 2 | 2025-09-12 | 2025-09-14 | 6 | 6 |  |
| R3 | 3 | 2025-09-19 | 2025-12-11 | 6 | 6 |  |
| R4 | 4 | 2025-09-27 | 2025-09-28 | 6 | 6 |  |
| R5 | 5 | 2025-10-03 | 2025-10-05 | 6 | 6 |  |
| R6 | 6 | 2025-10-12 | 2025-10-12 | 6 | 6 |  |
| R7 | 7 | 2025-11-01 | 2025-11-02 | 6 | 6 |  |
| R8 | 8 | 2025-11-08 | 2025-11-09 | 6 | 6 |  |
| R9 | 9 | 2025-11-15 | 2025-11-16 | 6 | 6 |  |
| R10 | 10 | 2025-12-06 | 2025-12-07 | 6 | 6 |  |
| R11 | 11 | 2025-12-13 | 2025-12-14 | 6 | 6 |  |
| R12 | 12 | 2026-01-10 | 2026-01-11 | 6 | 6 |  |
| R13 | 13 | 2026-01-23 | 2026-01-25 | 6 | 6 |  |
| R14 | 14 | 2026-02-01 | 2026-04-29 | 6 | 6 |  |
| R15 | 15 | 2026-02-07 | 2026-02-08 | 6 | 6 |  |
| R16 | 16 | 2026-02-13 | 2026-05-06 | 6 | 6 |  |
| R17 | 17 | 2026-03-15 | 2026-03-18 | 6 | 6 |  |
| R18 | 18 | 2026-03-21 | 2026-03-22 | 6 | 6 |  |
| R19 | 19 | 2026-03-28 | 2026-03-29 | 6 | 6 |  |
| R20 | 20 | 2026-04-25 | 2026-05-09 | 6 | 6 |  |
| R21 | 21 | 2026-05-02 | 2026-05-13 | 6 | 6 |  |
| R22 | 22 | 2026-05-16 | 2026-05-16 | 6 | 6 |  |

## Manifest Verification Findings

- Ready for replay: yes for Matchweeks 2-22
- Manifest weeks missing from data: none
- Data rounds missing from manifest: none
- Date mismatches: none after aligning `dashboard/matchweek_manifest.py` to the data-derived windows above.
- Fixture-count mismatches: none
- Unverified manifest windows: none for Matchweeks 2-22
- Week 1 historical-prior required: True
- Rescheduled/long-window rounds flagged in the manifest: R3, R14, R16, R20, R21

## Week-By-Week Replay Steps

1. Optionally rerun `python scripts/inspect_matchweek_windows.py --output reports/replay_manifest_check.md` before operational replay to confirm live data has not changed.
2. Start the API with `python -m uvicorn api.main:app --env-file .env`.
3. Start Streamlit with `PREDICTION_API_BASE_URL=http://localhost:8000 API_KEY=$API_KEY python -m streamlit run dashboard/app.py`.
4. In the dashboard, generate predictions for Matchweeks 2 through 22 in order.
5. Confirm each run appears in prediction history and therefore in `prediction_runs`.
6. Treat R3, R14, R16, R20, and R21 carefully because their windows include postponed/rescheduled fixtures.
7. Run evaluation after predictions are logged, using the dashboard-generated command or `python -m evaluation.run_evaluation --start-date <week_predict_from> --persist --run-trigger dashboard-season-2025-26-week-XX-evaluation`.
