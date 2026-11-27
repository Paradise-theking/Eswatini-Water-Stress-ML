# Paper 2 — Exploratory multi-state drought transition forecasting

## Purpose
Explore forecasting the hydroclimatic state several months ahead rather than reducing the target to a single dry/not-dry label. This is a complementary experiment to the onset and drought-state binary models; it does not replace the continuous SWBA forecast.

## State definitions
- Normal: SWBA > -0.5
- Watch: -1.0 < SWBA <= -0.5
- Drought: SWBA <= -1.0

These three states are mutually exclusive and ordered. Thresholds are exploratory operational groupings, not claims of validated impact thresholds.

## Targets
Predict the state observed at t+1, t+3, and t+6 months from information available at forecast origin t. The forecast origin features include contemporaneous climate variables, lagged climate features, SWBA at origin and lags, and month-of-year cyclical encodings.

## Evaluation
Expanding chronological folds. Model: balanced multinomial logistic regression (C=1). Baseline: training-fold, month-of-year state distribution with Laplace smoothing across the three classes. Scores include accuracy, macro-F1, multiclass log-loss and multiclass Brier score. Log-loss and Brier score are probability-sensitive and therefore important; accuracy alone is not sufficient.

## Cautions
- State counts are imbalanced; the drought class may be poorly represented in some folds.
- A fold with a single class in training cannot fit the multiclass model and is explicitly marked skipped.
- The baseline and model are assessed on identical fold periods.
- The experiment is exploratory and uses no frozen test-period tuning.
- This experiment is not an operational warning product and requires further validation.

## Script and outputs
Script: `scripts/paper2_transition_forecasting.py`
Outputs (after execution): `outputs/research/paper2_transition_forecasting/fold_scores.csv`, `summary.csv`, and `metadata.json`.
