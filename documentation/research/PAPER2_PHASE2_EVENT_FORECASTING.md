# Paper 2 — Phase 2: prospective event forecasting

## Experiment objective
Test whether drought-state and drought-onset probabilities at 1-, 3-, and 6-month horizons can be estimated using information available at the forecast origin, rather than repeating descriptive event inventory work.

## Tasks
- Future drought state: SWBA at origin+h <= -1.0.
- Future onset: SWBA at origin+h <= -1.0 AND the immediately preceding calendar month is > -1.0. Intermediate SWBA is used to construct labels only, never as forecast-origin features.
- Horizons: 1, 3, and 6 months.

## Predictor design
At forecast origin, meteorological variables from the expanded source, selected lagged meteorological variables, SWBA at origin and lags 1/3/6, and cyclical month encodings. No post-origin values are predictors. Median imputation and scaling are fitted within each training fold.

## Validation
Expanding chronological folds; never shuffle. Initial training period is at least 120 monthly records; validation blocks are 24 months where possible. Balanced logistic regression (C=1) and probability threshold 0.5 are fixed in advance. Metrics: PR-AUC, ROC-AUC (when defined), Brier score, precision, recall and F1. Fold event counts are reported. No threshold tuning or use of the frozen 2020–2025 test period.

## Interpretation caution
This is a baseline experiment, not a claim of improved skill. Small event counts may produce undefined or unstable metrics. Results are validation-only and will inform a later model-design stage. The severe threshold is not modeled in this phase because its scarcity is already established.

## Reproducible command
`python -m scripts.paper2_event_forecasting`
