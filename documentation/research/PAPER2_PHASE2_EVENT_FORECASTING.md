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
Expanding chronological folds; never shuffle. The initial training portion is at least 120 months and is approximately 60% of the available labeled timeline; the remainder is split into sequential validation blocks. Balanced logistic regression (C=1) and a decision threshold of 0.5 are fixed in advance. Metrics: PR-AUC, ROC-AUC (when defined), Brier score, precision, recall and F1. Fold event counts and skipped single-class folds are recorded.

## Baselines
Two fold-local probability baselines are included for comparison: (1) the training-fold event prevalence as a constant probability, and (2) a month-of-year event-rate estimate with Laplace smoothing (one pseudo-event and one pseudo-non-event), computed only from the training fold. These baselines avoid using validation labels to estimate forecast probabilities. The baseline comparison is a central part of interpretation; the classifier is not presumed superior.

## Interpretation caution
This is a baseline experiment, not a claim of improved skill. Small event counts may produce undefined or unstable metrics. Results are validation-only and will inform a later model-design stage. The severe threshold is not modeled in this phase because its scarcity is already established.

## Reproducible command
`python -m scripts.paper2_event_forecasting`
