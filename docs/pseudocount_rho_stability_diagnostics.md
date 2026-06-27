# Pseudocount and Rho Stability Diagnostics

Date: 2026-06-27

## Scope

This branch is intentionally narrow. It follows the champion-family calibration work and focuses only on:

- xG pseudocount stability around the current champion default;
- fitted Dixon-Coles rho stability and boundary behavior;
- at most one combined diagnostic if pseudocount stability supports it.

Out of scope:

- time-decay tuning beyond prior context;
- neural networks, PyTorch, embeddings, or deeper models;
- market odds as model features;
- broad hyperparameter grids;
- unrelated candidate families;
- production promotion decisions.

The operational champion remains frozen. Do not modify `models/champion_dc_xg.py` or `model/wsl_xg_model.py`.

## Current Evidence

The previous champion-family run used the shared 2025-10-01 to 2026-05-16 window with 19 weekly rolling folds and 109 evaluated fixtures.

Pseudocount evidence:

| xg_pseudocount | model | log_loss | brier_score | accuracy | avg_max_probability |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.05 | `champion_dc_xg` | 0.8669 | 0.5052 | 0.6239 | 0.5784 |
| 0.10 | `txg_xg_pseudocount_010` | 0.8649 | 0.5038 | 0.6330 | 0.5751 |
| 0.15 | `txg_xg_pseudocount_015` | 0.8638 | 0.5030 | 0.6147 | 0.5729 |
| 0.20 | `txg_xg_pseudocount_020` | 0.8636 | 0.5026 | 0.6147 | 0.5714 |

The probability metrics improved smoothly through `0.20`, while confidence softened. The open question is whether `0.20` is stable across folds or just helped one fixture cluster. `0.25` is included only as a saturation check, not as a broad search.

Fitted-rho evidence:

- `dc_fit_rho_each_fold` improved over champion on 12 of 19 folds by log loss and Brier.
- It hit the current rho grid maximum in 8 folds.
- Rho ranged from `-0.24` to `0.01`, with mean `-0.0389`, median `0.0000`, and standard deviation `0.0738`.

The open question is whether fitted rho is a useful shadow candidate or a noisy diagnostic whose gains rely on boundary behavior.

## Diagnostics To Run

Pseudocount candidates:

- `0.05` champion/default;
- `0.10`;
- `0.15`;
- `0.20`;
- `0.25` saturation check.

For each row, report aggregate log loss, Brier, accuracy, average max probability, probability entropy, high-confidence wrong count, actual-outcome probability below 5% and 10%, draw recall, draw log loss, fold-level deltas versus `0.05`, and fold win/loss counts.

Rho candidates:

- `champion_dc_xg` fixed rho `-0.13`;
- `dc_rho_mild_minus_08`;
- `dc_fit_rho_each_fold` with the current grid;
- optional `dc_fit_rho_each_fold_wide_grid`, labelled diagnostic only, if it can be implemented without touching the champion.

For fitted rho, report rho per fold, train size, test size, grid min/max, boundary hit counts, fold metrics, fold deltas, and whether boundary hits cluster in small or specific folds.

Combined diagnostic:

- Run exactly one combined fitted-rho plus best stable pseudocount diagnostic only if `0.20` or `0.25` is stable enough to carry forward.

## Shadow Criteria

A candidate can be carried forward as shadow-ready only if it:

- beats champion/default on aggregate log loss or Brier;
- improves across many folds, not just one or two;
- does not introduce worse overconfidence;
- does not materially damage draw handling;
- has a small, interpretable mechanism.

## Parking Criteria

Park a candidate if:

- gains reverse at `0.25` or are concentrated in a few folds;
- draw recall or draw log loss materially degrades;
- fitted rho keeps hitting the boundary or moves further into the boundary under a wider grid;
- the combined diagnostic stacks instability without clear probability-quality gains;
- results require broad tuning to look convincing.

No candidate should be promoted from this branch alone.
