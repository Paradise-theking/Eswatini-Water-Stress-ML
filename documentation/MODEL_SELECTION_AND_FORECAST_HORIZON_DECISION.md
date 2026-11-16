# WaterWatch model-selection and forecast-horizon decision

**Status:** Research decision record; no model promotion authorised  
**Date:** 2026-11-16  
**Purpose:** Keep the live prototype's model choice separate from the expanded retrospective model comparison and from the operational warning decision.

## 1. Current production candidate

The existing live API loads `models/water_stress_ridge.joblib`, a 15-feature preprocessing/model pipeline using median imputation, standardization and Ridge regression (alpha 0.05). It is a one-month-ahead SWBA regression under the original short-record climatology and feature specification.

The prototype-readiness branch now requires the target month to be wholly in the future at issue time. The observed common CHIRPS/ERA5-Land cutoff on 2026-11-07 was 2026-09-30, 38 days old at retrieval. The associated shadow audit showed that the October 2026 forecast was created 36.7 days after October began. That is retrospective for the declared target and cannot be counted as an advance warning.

**Decision:** Do not relax the issue-time guard. The current live one-month Ridge forecast is not presently demonstrated to be issuable before its target period starts.

## 2. Expanded continuous-regression evidence

The expanded historical experiment uses a longer 1981-2025 series and a different three-month SWBA target/reference construction. Its committed final-test metrics are:

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Lasso | 0.5089 | 0.7001 | 0.5155 |
| Ridge | 0.5077 | 0.7123 | 0.4985 |
| Tuned XGBoost | 0.5420 | 0.7465 | 0.4491 |
| Persistence | 0.6952 | 0.9833 | 0.0443 |
| Seasonal climatology | 0.7970 | 1.1294 | -0.2608 |
| Training-mean baseline | 0.8040 | 1.1435 | -0.2924 |

The rolling-origin summary reports:

| Model | Mean MAE | Mean RMSE | Mean R² |
|---|---:|---:|---:|
| Lasso / ElasticNet_Lasso | 0.3966 | 0.5253 | 0.5804 |
| Ridge | 0.4106 | 0.5304 | 0.5710 |
| Tuned XGBoost | 0.4275 | 0.5476 | 0.5443 |
| Persistence | 0.4719 | 0.6594 | 0.3404 |

### Interpretation

- Lasso has the lowest mean rolling-origin MAE/RMSE and the highest mean R² in the committed rolling-origin summary.
- On the final test, Ridge has a marginally lower MAE than Lasso, while Lasso has lower RMSE and higher R². The differences between Lasso and Ridge are small.
- Both linear candidates outperform the persistence and seasonal-climatology baselines on these reported splits. This is useful evidence that the expanded predictors contain signal beyond those baselines.
- XGBoost does not outperform the linear candidates in these results. There is no reason to add its operational complexity at this stage.
- The final-test and rolling-origin values are retrospective model-evaluation results. They do not establish that the inputs were accessible at the forecast issue time, that a user could have received the forecast before the target period, or that the model predicts reservoir storage, supply continuity or impacts.

**Decision:** Keep expanded Lasso and Ridge as the two leading research candidates. Keep XGBoost as a benchmark, not the preferred deployment candidate. Do not promote either linear candidate until the horizon and issue-time audit below passes.

## 3. Required forecast-horizon audit before model promotion

The feature/target-generation code and saved prediction rows must be inspected together. For every test prediction, reconstruct:

- forecast origin month (the latest month of information permitted at issue time);
- latest observation date for every input source;
- source retrieval timestamp and, where available, first-publication/vintage timestamp;
- exact target variable and its target month or aggregation window;
- earliest time the full target period begins and ends;
- forecast issue timestamp;
- whether the full target period was still in the future at issue time;
- whether any rolling feature, climatology, scaler, imputer, feature selector or tuning step used information after the origin;
- model artifact and input dataset hashes.

A row is eligible for prospective evaluation only if all model inputs were available at the issue time and the forecast was issued before the target period began. If the target is an accumulated three-month index, state whether the forecast concerns the next three complete months, a rolling window ending three months ahead, or a value aligned to the origin month. The variable name alone is not sufficient to establish lead time.

The audit must use date-specific archived source snapshots. A retrospective dataset refreshed after the target period is not proof that the values were available at the historical issue date.

## 3A. Critical clarification: the expanded model is not a three-month-lead forecast

Inspection of `scripts/build_expanded_forecasting_dataset.py` confirms:

- `forecast_origin_month = month_date(t)`
- `target_month = month_date(t+1)`
- `target_swba_3month = swba_3month(t+1)`
- predictors use lag 0 and earlier values, so the intended information origin is month `t`.

Therefore, the expanded Lasso/Ridge/XGBoost experiment predicts the next month's **three-month rolling SWBA**. It is a one-month lead forecast of a three-month accumulation/index, not a forecast three months into the future. The rolling three-month target window ends in the target month; it does not represent three future months beyond the origin.

This changes the proposed path: the expanded Lasso/Ridge results do not solve the one-month source-latency problem by themselves. Both the original and expanded experiments have a next-calendar-month target. The expanded dataset is useful because it has a longer historical record and a different target/reference construction, not because it has a longer lead time.

The existing test prediction CSV contains `target_month` and predictions but not the explicit origin or actual issue timestamp. The origin can be reconstructed as target month minus one month from the dataset construction, but actual historical source publication time is still unknown. Thus the backtest demonstrates a structural one-month forecast design, not proof that the inputs were available before each target month began.

## 3B. New true-lead experiment

The new `scripts/evaluate_true_lead_horizons.py` builds separate 1-, 2- and 3-month target leads from the expanded monthly SWBA source. For each origin month `t`, it uses predictors available at `t` and predicts the three-month rolling SWBA at `t+lead`. It applies the existing final-evaluation Lasso, Ridge and tuned XGBoost configurations where available, plus persistence, seasonal-climatology and training-mean baselines. It writes row-level predictions with origin month, target month, lead and input cutoff, plus horizon-specific metrics.

This is a candidate-screening experiment, not a replacement model or prospective validation. Hyperparameters are held fixed from the existing final evaluation; no per-horizon tuning is performed. The output must be run against the locally prepared monthly SWBA source and reviewed before choosing a lead. The 2- and 3-month horizons are only useful operationally if their inputs are actually available at issue time and the accuracy remains acceptable.

## 4. Model promotion gates

A candidate may enter a prospective shadow trial only after:

1. The target definition and forecast origin are unambiguous.
2. The target period is wholly in the future at issue time.
3. Feature construction is reproducible from origin-time inputs only.
4. The full preprocessing pipeline and model artifact are frozen and hashed.
5. A simple persistence and seasonal baseline are evaluated on exactly the same eligible origins.
6. Error metrics are reported by season and target severity, with uncertainty and sample counts.
7. Input data versions and issue-time forecast records are archived.

A shadow trial is not an operational alert. It is a period of prospective recording and later scoring with no automatic action.

## 5. Product implications for PachiPanda

The credible prototype should demonstrate:

- observed hydroclimatic conditions with source dates and data-quality status;
- a forecast only when its target horizon is genuinely ahead of issue time;
- model comparison transparently labelled as retrospective research;
- a durable archive of issue-time inputs, model versions and predictions;
- a separate operational-risk layer that remains inactive until water-operator data and decision thresholds are agreed.

The Sentinel-2 surface-water area and 30-day climate-balance indicators remain separate contextual proxies. They must not be combined into a reservoir-storage estimate. The current evidence does not include a verified reservoir stage/storage series, inflow/release series, or service-impact labels.

## 6. Immediate next experiment

Create a reproducible target/origin audit for the expanded three-month Lasso and Ridge runs using the saved target series and prediction-level outputs. Do not retrain or replace the live artifact during this audit. The output should be a row-level eligibility table, a summary of eligible versus ineligible predictions, and revised metrics calculated only on issue-time-valid rows.

If no saved prediction-level file contains enough origin metadata, regenerate the historical backtest from frozen source snapshots and explicitly label it as a reconstruction, not observed prospective performance.
