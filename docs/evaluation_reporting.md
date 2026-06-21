# Evaluation Reporting

The evaluation reporting layer turns prediction-vs-result rows into model
selection artefacts: metric tables, calibration summaries, and failure-analysis
sections.

## Probability Metrics

Log loss and Brier score matter because WSL predictions are probabilistic. They
reward models that put high probability on what actually happened and penalize
overconfident misses.

Log loss is especially sensitive to confident wrong predictions. If a model
assigns very little probability to the actual outcome, the penalty is large.
Brier score measures squared distance between the predicted three-way
probability vector and the actual one-hot outcome.

Accuracy alone is not enough. A model can pick the most likely outcome often
while still being badly calibrated or assigning poor probabilities. For betting,
forecasting, and model comparison, probability quality is the product.

## Calibration

Calibration asks whether predicted confidence matches observed frequency. If a
model's 60% predictions win about 60% of the time, that confidence band is well
calibrated. If they only win 40%, the model is overconfident. If they win 75%,
it is underconfident.

Confidence buckets summarize low, medium, and high confidence predictions.
Sparse buckets are flagged because WSL samples are small and a few matches can
move observed accuracy sharply.

## Failure Analysis

Failure analysis highlights the matches that explain aggregate metrics:

- worst misses by log loss or low probability assigned to the actual outcome
- best high-confidence correct calls
- breakdowns by predicted favourite and confidence bucket

These views help distinguish bad luck from systematic model weaknesses, such
as overrating home favourites or being too confident in draw probabilities.

## Champion and Challengers

The report helpers work for champion-only outputs today and multiple models in
future phases. When challenger models are added, their backtest predictions can
be combined with champion predictions and ranked on the same metric suite.
That supports model selection with comparable evidence instead of isolated
one-off experiments.

