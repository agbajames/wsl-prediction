# Neural Network Proof Of Concept

`neural_network` is a Phase 8C research-only challenger. It exists for roadmap
completeness and curiosity, not because one WSL season is enough evidence for a
neural model promotion.

## Model

The model is a tiny tabular MLP implemented with NumPy:

- one hidden layer
- 8 hidden units
- tanh activation
- L2 regularisation
- deterministic random seed
- fixed iteration cap

No PyTorch, TensorFlow, or new dependency is required.

## Features And Leakage Control

The model reuses the `xg` feature group from `improved_logistic_regression`.
Training features are generated sequentially inside each fold, so each match
only sees prior matches. The scaler is fitted on the training matrix only and is
then applied to prediction rows.

## Interpretation

This is evaluation-only. A poor result is evidence that the current dataset is
too small for neural models. A good result would still require shadow testing,
more data, and careful stability review before any production decision.

`champion_dc_xg` remains the operational/reference model.
