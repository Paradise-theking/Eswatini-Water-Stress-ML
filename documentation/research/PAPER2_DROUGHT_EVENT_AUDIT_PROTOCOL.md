# Paper 2 — Drought Event Audit Protocol (Phase 1)

## Purpose
Isolated research audit for event-focused SWBA forecasting. This branch does not modify Paper 1, production code, live inference, or app deployment.

## Source and reproducibility gate
Use the expanded monthly SWBA source: data/processed/expanded/mnjoli_water_stress_monthly_1981_2025_swba.csv. Confirm file access, row count, date continuity, duplicate months, missingness, SWBA definition, and rolling-window warm-up. Record a checksum and audit outputs. Do not silently impute or drop records; document all exclusions.

## Candidate event definitions (sensitivity analysis required)
- Dry state: SWBA <= -1.0.
- Severe dry state: SWBA <= -2.0.
- Dry episode: maximal contiguous run of dry-state months; retain single-month episodes.
- Onset: target month is dry (SWBA <= -1.0) and immediately preceding calendar month is not dry (SWBA > -1.0). Missing prior-month SWBA means unknown label, not negative.
- Severe onset: target month <= -2.0 and preceding month > -2.0, with missing prior data treated as unknown.
- Intensification: within an existing episode, a decline in SWBA relative to the immediately preceding month; exploratory. Separately track crossing into severe state (<= -2.0).
- Recovery: first month after an episode with SWBA > -1.0.

## Required event inventory
1. Counts of dry and severe months.
2. Episode count, duration distribution, peak intensity (minimum SWBA), cumulative deficit (definition to be explicitly stated), dates, season.
3. Onset counts by year/season, severe-state transitions, isolated events.
4. Boundary and missing-data cases.
5. Threshold sensitivity at SWBA <= -0.5, -1.0, -1.5, and -2.0. These are research thresholds, not validated impact thresholds.

## Prediction-target alignment
For forecast origin t and lead h in {1,2,3}, classification target uses SWBA at t+h. Predictors must be available by origin t. Onset at t+h requires the preceding calendar month's SWBA to define transition. For lead 1, this prior month is typically available at t; at leads 2 and 3 it is not known at issuance. Onset labels at longer leads are constructed retrospectively, without using intermediate SWBA as predictors. No target/intermediate outcome leakage.

## Modeling safeguards
- Assess class balance and event count before fitting.
- No oversampling, undersampling, threshold tuning, calibration, or hyperparameter selection using final test.
- Use chronological rolling-origin validation; preserve prior 2020–2025 frozen-model test results as historical benchmark, not a tuning set.
- Report precision, recall, F1, PR-AUC, Brier/calibration where feasible, false alarms/year, event detection and lead; uncertainty and limitations.
- Include simple baselines and disclose availability constraints.
- Do not equate SWBA drought labels with observed reservoir, supply, agricultural or impact drought without independent outcomes.

## Phase 1 deliverable
Reproducible descriptive audit and event inventory. Modeling starts only after audit and validation design review.
