# Paper 2 matched-origin diagnostic protocol

## Purpose
Reconstruct validation-level forecasts across four task families to inspect errors and class-specific behavior rather than relying on aggregate fold scores.

## Forecast origin and features
At origin month t, the predictors include meteorological/hydroclimatic observations available for t, lagged predictor values, observed SWBA at t, its historical lags, state at t, and month-of-year harmonics. Targets refer only to future observations t+h. This is an origin-time forecast design; availability of current-month observations must be explicit in any operational interpretation (e.g., forecast issued after month t is observed).

## Targets
- Binary drought state: future SWBA <= -1.
- Onset: future month t+h is <= -1 while month t+h-1 is above -1.
- Multistate severity: Normal (SWBA > -0.5), Watch (-1 < SWBA <= -0.5), Drought (SWBA <= -1).
- Direction: sign of future severity-state index minus current-state index, where indices are Normal=0, Watch=1, Drought=2. The numerical direction labels are -1/0/+1.

## Validation
A common full-timeline split is applied: first 60% of monthly rows for initial training, then five sequential validation blocks across the remainder. Each fold uses the expanding training history before its validation block. The target rows are filtered for available labels. Missing values are imputed and standardized within the model pipeline fitted on training rows only. Logistic Regression uses class-weighting; probability estimates should not be assumed calibrated.

## Metrics
Binary: TP/FP/FN/TN, precision, recall, F1, false-positive rate, average precision (PR-AUC), and Brier score. Multiclass: accuracy, macro-F1, log-loss and Brier score. Fold summaries should be accompanied by fold-level counts and prediction files.

## Limitations and audit checks
1. This is exploratory validation, not final held-out testing.
2. Sparse onset counts limit stability of metrics.
3. Class-weighted probabilities may distort event prevalence.
4. Predictor availability must be operationally specified. Unlagged t climate values are valid only if the system issues the forecast after month t data are available.
5. The state target and directional target represent different questions; avoid comparing their scores as if identical.
6. Missing classes in individual training folds and probability-column mappings should be explicitly audited.
7. Comparisons against baselines should be rerun on identical validation origins and target support before interpreting improvements.

## Intended output
Prediction-level records, fold diagnostics, summary metrics and experiment metadata. The final model and final evaluation period remain unselected.
