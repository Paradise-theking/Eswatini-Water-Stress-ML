# Paper 2 — Cumulative drought-onset forecasting protocol

## Research question
Does a hydroclimatic logistic event-probability model provide information about the occurrence of at least one new drought onset during a forecast window beyond a conditional historical state-transition benchmark?

## Target
For origin month t and horizon h in {1, 3, 6}, the cumulative target is one if any month j in t+1,...,t+h satisfies SWBA_j <= -1 and SWBA_(j-1) > -1. Otherwise it is zero. The origin month is included only as the predecessor for an onset in t+1; it is not counted as a future event.

The target remains missing unless every SWBA label from t through t+h is observed. In particular, missing future SWBA values are never treated as non-events.

## Validation and matching
Use the established five-fold expanding chronological validation: the first 60% of the complete monthly timeline is the initial training period and the remaining 40% is divided into five contiguous validation blocks. For horizon h, purge training origins unless origin+h is strictly earlier than the first validation origin. All model variants and the conditional transition benchmark are evaluated on identical valid validation origins within each fold.

## Models
1. Logistic regression without class weighting.
2. Logistic regression with balanced class weights.
3. Conditional origin-state transition benchmark: estimate cumulative-onset frequency in the training fold conditional on the observed origin severity state (normal, watch, drought), with shrinkage toward the fold event prevalence.
4. Training-fold prevalence benchmark.

Imputation and scaling are fitted within each logistic model's training fold. The transition baseline uses training-fold labels only.

## Evaluation
Report pooled out-of-fold and fold-specific PR-AUC, ROC-AUC when defined, Brier score, log loss, precision, recall, F1, confusion counts and false alarms per 100 forecast origins. Report sensitivity at fixed probability thresholds 0.10–0.50; do not choose an operational threshold on the validation results and present it as independently validated.

## Interpretation
Cumulative windows overlap, and drought onsets are rare. Pooled observations are therefore not independent. Include fold-level event counts and uncertainty intervals in subsequent analysis. This experiment is validation-only and does not use the independent final test period. It is a comparison of predictive probability information, not yet a full operational warning-system evaluation.
