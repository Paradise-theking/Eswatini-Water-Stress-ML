# Paper 2 — Directional drought-transition forecasting

## Research question
Can available climate and catchment-state predictors at origin t anticipate whether the discretized SWBA state will improve, remain unchanged, or deteriorate by t+h?

## State encoding
Normal=0 (SWBA > -0.5), Watch=1 (-1.0 < SWBA <= -0.5), Drought=2 (SWBA <= -1.0). Direction is the future state minus the origin state: negative is improvement, zero is unchanged, positive is deterioration. This directional target deliberately collapses transition magnitudes and should be considered complementary to the full-state and onset experiments.

## Forecast horizons
1, 3 and 6 months. Predictors include climate variables available at origin, selected lags, SWBA history and origin state, and month-of-year encodings.

## Models and evaluation
- Balanced multinomial logistic regression (C=1)
- Month-of-year directional-frequency baseline with Laplace smoothing, learned only from each training fold

Expanding chronological folds. Metrics: accuracy, macro-F1, multiclass log-loss and multiclass Brier score. Class counts are recorded. Single-class training folds are explicitly flagged.

## Interpretation caveats
This is exploratory. Direction is not a full transition matrix: it treats normal-to-watch and watch-to-drought as deterioration, for example, without distinguishing transition pathways. Sparse transitions and changing climate regimes may destabilize estimates. No final test period is used for tuning. Results must be compared with binary onset/state and full three-state forecasts before conclusions.

Script: `scripts/paper2_direction_forecasting.py`
Outputs after execution: `outputs/research/paper2_direction_forecasting/fold_scores.csv`, `summary.csv`, and `metadata.json`.
