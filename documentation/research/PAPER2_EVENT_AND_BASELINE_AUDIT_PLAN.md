# Paper 2: event-based and baseline audit plan

## Rationale
The next analysis evaluates whether warning performance is useful at event level and whether the apparent model benefit persists against non-ML benchmarks. Aggregate metrics can hide false alarm burden and event misses.

## Planned analysis (after matched-origin outputs are verified)
1. Event inventory at SWBA <= -1: contiguous episodes, onset/end, duration, minimum severity and cumulative deficit (sum of absolute SWBA below threshold).
2. Matched prediction event audit: within each validation fold and horizon, classify onset hit/miss, lead-time category (only if supported by prediction origin), false alert, and event coverage. Avoid treating multiple forecasts for one episode as independent events.
3. Baselines on identical origins:
   - persistence: origin severity/state carries forward to future horizon;
   - seasonal-state baseline: same-month historical transition frequency / class distribution, fit on training folds only;
   - conditional transition matrix: origin state transition frequencies, Laplace smoothing, fold-local;
   - model forecasts.
4. Decision analysis: precision, recall, false alarms per validation year, event hit fraction, PR-AUC and Brier. Report sample and event denominators.
5. Calibration: evaluate raw probabilities with reliability bins, Brier score and calibration slope/intercept where sample sizes permit. Do not rely on class-weighted logistic probabilities as calibrated probabilities. Test calibration only inside training/validation logic without contaminating folds.
6. Sensitivity to severity threshold: exploratory thresholds -0.5, -1, -1.5, -2, keeping the primary warning threshold declared in advance and noting extreme event sparsity.

## Event-level caution
A drought episode is a contiguous sequence below the chosen threshold. Forecasts within the same episode are not independent event detections. Event-level evaluation must aggregate forecasts per episode and define a hit rule before inspecting results (e.g., an onset warning within a fixed lead window), then separately report false alerts during non-event periods.

## Proposed paper contribution
An interpretable, state-aware probabilistic drought-transition and deterioration forecasting framework evaluated against persistence and empirical state-transition baselines, with emphasis on event-level warning utility rather than only continuous-index prediction.

This is a hypothesis for investigation, not an established finding. Model selection, hit rules, calibration, and thresholds are not finalized.
