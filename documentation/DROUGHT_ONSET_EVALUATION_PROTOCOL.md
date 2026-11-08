# Drought-Onset Model Evaluation Protocol

## Purpose and scope

This protocol governs research evaluation of monthly drought-onset warnings for
the Mnjoli W60E catchment. It does not change or validate the Ridge model used by
the live forecast API. Continuous SWBA regression and drought-onset
classification are separate prediction tasks and must be reported separately.

The existing 2006-2025 chronological folds have already been examined during
model development. They are retrospective development backtests, not an
untouched final test set. No result from those folds alone supports an
operational drought-warning claim.

## Frozen index and event definition

- Index: three-month accumulated precipitation minus PET, standardized by
  calendar month.
- Reference climatology: 1981-2010, calculated from the three-month water
  balance available in each calendar month.
- Standard deviation: sample standard deviation (`ddof=1`), matching the
  expanded SWBA construction.
- Drought month: `SWBA <= -1.0`.
- Drought onset: the first drought month immediately following a valid
  calendar month with `SWBA > -1.0`.
- Eligibility: a forecast origin must itself have `SWBA > -1.0`; this measures
  advance warning rather than detecting an already active drought.
- Horizons: one month is primary. Two- and three-month leads are exploratory
  and must not be presented as reliable unless independently supported.

The separate 2015-2021 climatology used by the original short-record Ridge
study defines a different index. Do not mix its values or performance metrics
with the expanded 1981-2010 SWBA analysis without explicitly reconstructing
both on a common definition.

## Data and label audit gate

The committed historical data currently ends in 2025. Refresh the common,
complete daily coverage from CHIRPS and ERA5-Land, append it to a separate
research dataset, and audit it with:

```powershell
.\.venv\Scripts\python.exe .\scripts\refresh_shadow_evaluation_data.py
```

The script uses the configured Earth Engine project and catchment, limits
updates to the latest complete month available in both sources, verifies
daily continuity and cross-source alignment, preserves the existing
1981-2025 series, reconstructs the fixed 1981-2010 SWBA, and invokes the
audit below. It writes versioned outputs under
`outputs/expanded/shadow_evaluation_data/`; it does not replace the canonical
historical files.

At the 2026-11-06 refresh, CHIRPS was available through 2026-09-30 and
ERA5-Land through 2026-10-26. Their latest common complete month was
2026-09. All 273 daily observations aligned for each source, all 540
historical months were preserved within `1e-8`, and the expanded 549-month
dataset passed every audit check. The audit found the same 30 onset events as
the historical version; no new onset at SWBA `<= -1.0` occurred through
2026-09. These refreshed values are still subject to the source products'
revision and reporting latency.

For a historical input or a later data refresh, the audit can also be run
directly from the repository root:

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_drought_onset_data.py
```

The audit writes `audit_report.json`, `onset_events.csv`,
`eligible_forecast_samples.csv`, and `chronological_fold_counts.csv` under
`outputs/expanded/drought_onset_model_preparation/data_audit/`. The report
records the input SHA-256, data coverage, reconstructed index comparisons,
onset counts, and fold sample/event counts.

The audit must pass before training. Investigate any mismatch in monthly
continuity, stored year/month fields, water-balance arithmetic, rolling
three-month totals, 1981-2010 climatology, SWBA reconstruction, or eligible
forecast-input completeness. Preserve the audit outputs with the model
evaluation so results can be tied to the exact input dataset.

### Verified audit

The current audit passed for 540 consecutive months from 1981 through 2025.
It found 30 onset events and 77 drought months. Reconstructed SWBA matched
the stored index within the `1e-8` tolerance (maximum absolute difference
`5.44e-15`), and there were no incomplete hydro-feature rows among eligible
forecast samples. The lead-adjusted one-month outer test blocks contain 5, 5,
2, and 1 onset events, respectively; the final block is especially
underpowered for onset evaluation. The report and supporting CSVs are in
`outputs/expanded/drought_onset_model_preparation/data_audit/`.

### Restart feasibility screen: 1- to 3-month onset windows

To guide a fresh model design without treating prior experiments as
confirmation, run the exploratory feasibility screen:

```powershell
.\.venv\Scripts\python.exe .\scripts\screen_drought_warning_feasibility.py
```

The script requires the current historical monthly input to match its passing
data-audit SHA-256. It compares a training-prevalence forecast, a smoothed
origin-calendar-month baseline, a one-feature current-SWBA logistic model, and
a fixed four-feature hydroclimate logistic model. The event outcome is at least
one onset during the next 1, 2, or 3 months, with origins required to be outside
drought. Training outcome windows are kept entirely before each chronological
test block; test origins are restricted so their complete outcome windows end
within that block. Calendar-month anomaly statistics and model preprocessing
are fit from the training period only.

The existing 2006-2025 folds have already informed model development, so this
is a bounded retrospective feasibility screen, not an independent test and
not a selection of a deployable model. It reports Brier skill and average
precision only; it does not select warning thresholds or calculate false
warning episodes. Overlapping lead windows are dependent and may count one
onset in several forecast origins. Input values are retrospective and do not
establish publication-time availability. Outputs, source hash, fold metrics,
and predictions are written to a new timestamped directory under
`outputs/expanded/drought_onset_model_preparation/`; no canonical data, model,
or alert behavior is changed.

### Warning-budget feasibility screen

After the probability screen, test whether fixed baseline and logistic
probabilities can meet the user-selected false-warning limits without tuning
against the test outcomes:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_warning_budget_feasibility.py
```

For each 1-, 2-, and 3-month horizon and each chronological test fold, the
script fits the same four fixed methods using only prior training windows. It
selects each method's threshold on the immediately preceding five-year
validation block, maximizing unique-onset event recall subject to no more than
0.5 false-warning episodes/year and 3 false-warning months/year, then evaluates
that threshold on the later test fold. A false-warning month is a warned
forecast origin whose forward window contains no onset. Consecutive warned
origins form one warning episode, which is false only when none of its forecast
windows contains an onset. A unique onset is counted as detected if at least
one alerted forecast window covers it.

These are retrospective development folds already examined during prior work,
so results are exploratory and cannot serve as independent validation.
Threshold selection on a five-year block with few events is unstable; forward
windows overlap, and one onset may be covered by multiple warnings. The test
fold reports both warning burden and unique event hits; passing a retrospective
fold is not evidence that the policy generalizes. Source availability at
forecast issue time remains unverified. Results and prediction-level data are
written to a new timestamped directory under
`outputs/expanded/drought_onset_model_preparation/`. The script does not
change the prospective shadow log, model, canonical data, or live alert
behavior.

#### Feasibility-screen result (2026-11-07)

Using the audited 1981-2025 input (SHA-256
`1abd8208657022a3a9b0cda93cb3b5c586fddda68676e650d3a1df74480b2bd0`), the
fixed four-feature hydroclimate model met both warning limits in all four
one-month test folds, but detected only 3 of 13 unique onset events. It
produced 7 false-warning months and 5 false-warning episodes across those
20 test-years. The one-month current-state model also detected 3/13, and met
both limits in 3/4 folds.

At two months, the hydroclimate model detected 8/13 events, but met both
limits in only 2/4 folds (11 false-warning months and 10 false-warning
episodes total). At three months, it detected 11/13, but also met both limits
in only 2/4 folds (16 false-warning months and 12 false-warning episodes).
These longer-horizon figures count each onset once even if multiple warnings
cover it; warning windows overlap, so they are not independent detections.

This shows a retrospective trade-off, not a deployable solution: the
one-month candidate stayed within the selected warning budget but had low
event sensitivity, while higher apparent detection at longer horizons did
not consistently stay within budget. The four test folds contain only 13
onset events and have already influenced development. Do not select a final
horizon or deploy an onset alert from these results. The detailed validation
thresholds, per-fold outcomes, and forecast-level records are in the timestamped
`warning_budget_feasibility_*` output directory.

### CHIRPS v2-to-v3 transition assessment

The existing historical and shadow precipitation series use the CHIRPS v2
Earth Engine collection `UCSB-CHG/CHIRPS/DAILY`. The latest recorded refresh
reaches 2026-09-30. Do not silently replace or splice those values with v3.

The CHC-linked `ropensci/chirps` client documents the final daily v3 GeoTIFF
archive at
`https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/{year}/`.
The CHC host timed out from this environment, but the public dynamical.org
catalog entry
<https://dynamical.org/catalog/ucsb-chc-chirps-analysis-final/> provides the
same final/rnl product in an Icechunk/Zarr mirror. The
mirror catalog verifies global land coverage, 0.05-degree resolution,
1981-to-present coverage, daily units of `kg m-2 s-1` (converted to mm/day),
and final-product release typically two to three weeks after month end. Its
daily distribution uses ERA5 to split pentadal CHIRPS totals. The mirror rounds
stored values to improve compression and is licensed CC BY 4.0; cite both the
CHC source and dynamical.org processing when using these extracted values.

Run the overlap workflow from the repository root with Python 3.12 or newer
(Icechunk v2 requires that runtime):

```powershell
py -3.14 -m venv .venv-chirps-v3
.\.venv-chirps-v3\Scripts\python.exe -m pip install -r .\requirements-chirps-v3-transition.txt
.\.venv-chirps-v3\Scripts\python.exe .\scripts\run_chirps_v3_transition_comparison.py
```

The script reads only the cells intersecting the saved Mnjoli catchment from
the public mirror. It hashes and records the catchment and v2 inputs, the
Icechunk snapshot, retrieval time, product and units; checks daily/monthly
completeness; and writes results to a new timestamped directory under
`outputs/expanded/chirps_v3_transition/`. It does not modify canonical data,
the live forecast, or alert behavior. It reports monthly and annual
precipitation differences, calendar-month summaries, SWBA and drought-label
changes under both a retained v2 reference and v3's own 1981-2010 reference,
and changed onset months. The v2 daily aggregation reproduces the existing
monthly precipitation to within `5.7e-14` mm and the stored SWBA within
`7.0e-15` in an offline reconstruction check.

The products share the same saved catchment polygon and native grid. To
quantify the reducer difference across the whole overlap, run:

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_chirps_v2_spatial_reducer.py
```

The audit samples all 11 native v2 cells intersecting the catchment for all
16,436 days, applies the same fractional cell-area weights as v3, and compares
the result to the archived Earth Engine catchment means (`mean`, scale 5566 m,
`bestEffort=True`). The area-weighted v2 total is 31,802.34 mm versus
31,836.12 mm from the existing reducer, a difference of `-33.77 mm`
(`-0.106%`) over 45 years. Daily correlation is `0.999989`, daily MAE is
`0.0086 mm`, and the maximum annual-total difference is `0.218%`; all 45
area-weighted annual totals are slightly lower. The full-period reducer
difference is therefore small relative to the v2/v3 precipitation shift.

Using this matched spatial reducer, the v3 total is `+19.36%` above v2
(versus `+19.23%` using the original v2 aggregation), and the SWBA drought
month counts and onset comparisons are unchanged at the reported precision.
This supports the conclusion that the large historical difference is not
explained by the catchment reducer. It does not isolate the CHIRPS version
change from v3-specific processing, daily disaggregation, and mirror rounding.
The complete cell weights, pixel-day samples, daily/monthly/annual comparisons,
and hashes are in
`outputs/expanded/chirps_v3_transition/run_through_2025_20261106T181031Z/v2_spatial_reducer_audit/`.

Preliminary v3 data are not used. The full final/rnl comparison was run for
1981-01 through 2025-12 (16,436 complete daily values and 540 complete
months). The v3 catchment total was 37,958.46 mm versus 31,836.12 mm for v2,
a difference of +6,122.34 mm (+19.23%). Monthly totals had correlation
`r=0.947`, mean bias `+11.34 mm/month`, MAE `15.16 mm`, and RMSE `24.90 mm`.
The mean v3 monthly total was higher in all 12 calendar months. Annual v3
totals were higher in 42 of 45 years, but lower in 2022-2024, so the
difference is not a constant offset.

The v2 index reconstruction passed its parity check against stored data
(maximum absolute SWBA difference `7.0e-15`). Over the 538 months with valid
three-month SWBA, v2 has 77 drought months and 30 onsets. Applying the same
1981-2010 reference statistics used by v2 to the v3 precipitation gives 54
drought months, 27 changed drought-month labels, and 25 onsets (14 v2 onset
months disappear and 9 appear). Recomputing the 1981-2010 reference
statistics from v3 gives 75 drought months, 24 changed drought-month labels,
and 32 onsets (7 v2 onset months disappear and 9 appear). These are changes
to the historical index and labels, not model skill or independent event
detection results. The precipitation difference, historical index impact,
and full onset-month lists are recorded in the generated CSVs and manifest.

Results and source hashes are in
`outputs/expanded/chirps_v3_transition/run_through_2025_20261106T181031Z/`.
The stored snapshot is `T5RTYW439H0E8VF37CQG`. The direct CHC host still timed
out from this environment; the data were read from the public dynamical.org
mirror. Since v3 final data are typically released two to three weeks after
month end, this final product alone does not satisfy timely early-warning
input requirements. Do not retrain the onset model, relabel history, splice
the series, or change live alerts based on this study alone. First harmonize
the v2/v3 native-grid reducers (or quantify their difference) before
attributing all observed differences to CHIRPS product revision.

The daily URL pattern and product distinction are documented in the
`ropensci/chirps` source:
<https://github.com/ropensci/chirps/blob/master/R/get_chirps_raw.R>.
Its README reports that CHIRPS v2 is scheduled to retire after the December
2026 release:
<https://github.com/ropensci/chirps/blob/master/README.md>.

## Retrospective development evaluation

Use forward-in-time rolling evaluation only. Keep the existing five-year test
blocks (2006-2010, 2011-2015, 2016-2020, and 2021-2025) as retrospective
outer backtests, with the preceding five years as a distinct validation block.
For lead `L`, training examples must have target labels observable before the
first forecast origin in the next block; enforce the lead-adjusted cutoff at
every training/validation/test boundary.

Because these blocks and their outcomes have already informed experiments,
their purpose is to measure temporal robustness and expose failure modes—not
to act as an independent final test. Do not choose a final model by selecting
the best score across these outer test results.

Within each outer development fold:

1. Perform feature and model selection, hyperparameter selection, and all
   preprocessing inside inner forward-chaining folds drawn only from the
   outer training period.
2. Fit the selected pipeline on the outer training period. Use the subsequent
   five-year validation block only to select the classification threshold.
   The initial warning policy prioritizes recall subject to a predeclared cap
   of 0.5 false-warning episodes per year and 3 false-warning months per year.
   If multiple thresholds meet both caps, choose the one with greatest recall,
   then precision, then fewer warnings. Record the validation counts and rates.
   Fit probability calibration only if event support permits a reliable
   estimate.
3. Freeze the complete pipeline, any calibration, and threshold before
   scoring the outer test block. Never refit, recalibrate, or retune using
   validation or outer-test outcomes after the threshold is selected.
4. Report every fold as well as pooled metrics. Include event counts and
   uncertainty; do not hide a fold with no detected events behind a pooled
   score.

Compare against training-prevalence and calendar-month baselines, plus a
predeclared simple current-state benchmark. Use the same eligible samples and
fold cutoffs for each baseline and candidate model.

Run the one-month evaluation from the repository root with:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_nested_drought_onset_backtest.py
```

The script requires a passing audit report whose input hash matches the
current dataset. The expanded candidate grid includes current SWBA state,
core hydroclimate predictors, state plus hydroclimate, state plus seasonality,
six-month water-balance memory with both soil-water layers, atmospheric
demand, and antecedent stress. These families were motivated by prior
exploratory event-control associations, not selected using the outer-test
scores. The script evaluates logistic regression across fixed regularization
and class-weight options. Inner folds are expanding, five-year validation
blocks beginning in 1991; candidates are ranked by mean Brier score, with
average precision and parsimony as tie-breakers. Calendar-month anomaly
references, median imputation, and model scaling are learned within each
corresponding training portion. The outer validation block selects the
threshold only; no model refit or probability calibration is performed there
because event counts are too small.

### One-month backtest and warning-budget sensitivity

The first nested run used F1 thresholds and produced 8 hits, 29 false-warning
months, and 5 misses across the 13 onsets in 209 eligible outer-test samples.
The warning budget was then set to prioritize recall while allowing at most
0.5 false-warning episodes and 3 false-warning months per year on the
validation block. Under this stricter, user-selected policy, the model
produced 3 hits, 12 false-warning months, and 10 misses (precision 0.200,
recall 0.231, F1 0.214) across the same test samples. The corresponding pooled
Brier score remained 0.0553 versus 0.0589 for the training-prevalence baseline
(Brier skill 0.061); threshold selection changes warning classifications,
not probability scores. The current-state logistic baseline had similar
classification performance (3 hits, 11 false-warning months, 10 misses) and
Brier skill 0.056.

The warning budget did not generalize consistently: false-warning rates
exceeded the caps in the 2011-2015 test block, while the selected thresholds
issued no warnings in the 2016-2025 blocks and missed all three onsets there.
Across the four tests, the model produced 5 false-warning episodes in total
(0.25 per year) and 12 false-warning months (0.60 per year). Year-block
bootstrap intervals remain broad: recall 0.000-0.471, precision 0.000-0.333,
F1 0.000-0.345, and Brier skill -0.199-0.310. The one-onset 2021-2025 fold
still cannot support a stable discrimination estimate.

An additional hindsight feasibility analysis searched each outer test fold
for its best threshold under both warning caps. This is an optimistic
diagnostic because it uses the test labels and is not a valid prospective
model result. Even with this advantage, only 3 of 13 onsets could be detected
across the four folds while staying within both caps fold-by-fold: 2/5 in
2006-2010, 0/5 in 2011-2015, 0/2 in 2016-2020, and 1/1 in 2021-2025. Thus
the current model scores do not show useful, consistent detection under the
selected warning budget.

#### Expanded antecedent-feature development check

The candidate grid was expanded to test the six-month water-balance,
second-layer soil-water, maximum-temperature, and solar-radiation signals.
This is another development search on the already-examined historical folds,
not independent validation. Three eligible input rows lack the added
six-month water-balance value; the model pipeline imputes such sparse values
using medians learned from each training fold, preserving the audited
eligible-sample cohort.

Before interpreting this search, a predictor-correlation audit on all 460
eligible forecast origins found that the calendar-month-standardized
three-month water balance and SWBA are perfectly correlated (`r = 1.000`):
they are two representations of the same index. Soil-water layers 1 and 2
were also highly correlated (`r = 0.916`), as were PET and maximum temperature
(`r = 0.800`). To avoid counting SWBA and its three-month water-balance
equivalent twice in the state-plus-hydro family, that candidate was corrected
to include SWBA, precipitation, PET, and layer-1 soil water. The candidate
grid still includes the broader, non-duplicate six-month memory and
atmospheric-demand families.

On the corrected nested run, inner selection chose state plus hydroclimate
for 2006-2010, current SWBA state for 2011-2015 and 2016-2020, and core
hydroclimate for 2021-2025. Pooled outer-test results were 5 of 13 onsets
detected, 14 false-warning months, and 8 misses (precision 0.263, recall
0.385, F1 0.313); Brier skill was 0.059 against training prevalence. The
current-state logistic baseline detected 3 onsets with 11 false-warning
months. However, the corrected model exceeded the false-warning episode cap
in the 2011-2015 test fold (4 episodes, 0.8/year), and the later fold has
only one onset. The modest pooled improvement is therefore not stable
evidence of warning skill.

The preceding exploratory matched-control summaries show large retrospective
contrasts for several antecedent variables, but their uncertainty estimates
must not be treated as confirmatory. Each onset was compared with all
seasonally eligible controls; onsets in the same target calendar month reuse
exactly the same control set (about 24 origins per onset on average). Thus
the event-level paired contrasts are dependent, and event bootstrap/Wilcoxon
uncertainty can be too optimistic. Multiple-testing adjustment does not
correct this control-set dependence. Treat these associations as hypotheses,
not independent evidence or a feature-ranking basis.

The event-level predictor extraction in
`scripts/audit_drought_onset_predictors.py` confirmed 30 onset rows and
recorded predictors from the preceding month. It checks historical values,
not when those values would actually have been published or finalized.
Operational forecast-time availability and revision latency remain
unverified; any future shadow forecast must log the data cutoff and the
timestamp when each source became available.

The corrected model run is saved under
`outputs/expanded/drought_onset_model_preparation/nested_chronological_backtest_decorrelated_features_1month/`.
The preceding expanded-grid run remains under
`outputs/expanded/drought_onset_model_preparation/nested_chronological_backtest_extended_features_1month/`;
the original four-family run remains under
`outputs/expanded/drought_onset_model_preparation/nested_chronological_backtest_1month/`.

### Untouched 2026 chronological holdout

The following holdout result is for the original, smaller candidate grid. The
feature-grid expansion and subsequent de-duplication above occurred after the
2026 outcomes had been inspected, so 2026 must not be used to validate or
select the revised candidate procedure. Do not rerun the updated search
against 2026 and call it an untouched test; evaluate the revised, frozen
procedure on future observed months instead.

The refreshed data through September 2026 was reserved from model fitting and
threshold selection. To evaluate it, the locked procedure selected its
candidate using expanding inner folds ending by 2020, fit the chosen model on
data through December 2020, and selected a threshold on 2021-2025 outcomes.
The selected candidate was the four hydroclimate predictors (`C=1`, no class
weight); its validation threshold was 0.3146. It predicted no warnings during
the nine eligible target months from January through September 2026. There
were also zero observed onsets, so there were no hits, misses, or false
warnings in this short holdout.

The 2026 result is genuinely out of sample with respect to the fitted model,
but it cannot verify drought-detection sensitivity: no positive event occurred.
Its low Brier score (0.00054) and positive skill against the training
prevalence baseline are not evidence of onset detection because all nine
outcomes were negative. The 2026 holdout is therefore useful only as an
initial no-event/false-alert check, not as confirmation that the warning model
works. Continue non-alerting shadow observation until enough onset events and
seasonal cycles accrue; do not treat the episode/month caps as demonstrated
annual rates from nine months.

These are retrospective development backtests on folds already examined in
prior experiments. The result is a research baseline, not an independent
confirmation or operational warning model. The observed performance under
the chosen warning budget is not sufficient for an onset-warning claim.
Detailed fold scores, baselines, predictions, episodes, reliability bins, and
bootstrap intervals are saved under the corresponding original and expanded
feature-grid result directories.
The post-hoc feasibility analysis and untouched 2026 predictions are saved
under
`outputs/expanded/shadow_evaluation_data/refresh_through_2026-09/unseen_2026_evaluation/`.

### Prospective non-alerting shadow log

The first non-alerting shadow forecast was archived on 2026-11-06. It uses
the exact frozen candidate from the corrected 2021-2025 chronological fold:
training through December 2015 (349 samples, 27 onsets), validation during
2016-2020, the four core hydroclimate predictors, logistic regression
(`C=1`, no class weighting), and the validation-selected threshold `0.256976`.
The 2021-2025 test outcomes were not used to fit or threshold this model.
Because this deliberately preserves that fold's training cutoff, this shadow
model is stale and is only a procedural/data-pipeline check; it is not being
presented as the best current model or as an operational recommendation.
Refitting with newer labels would require a separately locked, prospectively
validated threshold and would change the model version.

The refreshed data were audited through September 2026. The model forecast
from September to October 2026 with probability `0.002038`, below the frozen
threshold. No operational alert was issued. October was not yet present as a
fully observed month in the refreshed series, so no outcome score has been
written. The row was created on November 6, after its October target month had
already started; it is therefore a pipeline exercise, not a timely prospective
forecast or evidence of sensitivity/model accuracy. The issue-time guard now
rejects forecasts when either the input snapshot or issue time is on/after the
target month's start.

Use the workflow from the repository root:

```powershell
.\.venv\Scripts\python.exe .\scripts\manage_drought_onset_shadow.py forecast
.\.venv\Scripts\python.exe .\scripts\manage_drought_onset_shadow.py score
```

`forecast` validates the historical and refreshed data hashes, checks that
the fitted model and threshold match the frozen fold artifacts, refuses to
forecast a target already present in the input, and appends one immutable row
to `outputs/expanded/drought_onset_model_preparation/prospective_shadow/shadow_forecasts.csv`.
It never sends an alert. `score` does nothing until the target month is in a
passing refresh audit and the manifest reports complete common coverage
through that month. It then appends the observed onset and warning outcome to
`shadow_scores.csv`, once per target month. When a later refresh is available,
pass its matching data, audit, and manifest paths explicitly to either mode.
By default, the shadow command resolves the latest refresh snapshot whose audit
passed. To reproduce a historical run, pass that run's matching paths
explicitly.

```powershell
.\.venv\Scripts\python.exe .\scripts\manage_drought_onset_shadow.py forecast
.\.venv\Scripts\python.exe .\scripts\manage_drought_onset_shadow.py score
```

The current log is deliberately limited to this one locked, older model. Do
not edit its threshold or retrain it in place; any new candidate must receive
a distinct version and start a separate prospective log. Source-product
availability and revisions remain limitations: the archive records refresh
time, common data cutoff, file hashes, model/config hashes, probability,
threshold, and forecast-origin values for later audit.

Refresh runs are now timestamped, append-only snapshots:
`refresh_through_YYYY-MM_YYYYMMDDTHHMMSSZ`. A rerun never overwrites an
earlier extraction, manifest, or audit. Repeated runs create observations for
estimating retrieval freshness; they still do not establish true first-
publication latency or source revisions unless upstream release/vintage
timestamps are available.

### Forecast issue-time and data freshness audit

Run the provenance audit after each refresh:

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_forecast_issue_time.py
```

It reports the age of each source's latest dated observation at retrieval and
checks archived shadow rows against the beginning of their declared target
month. These age values are retrieval-time freshness measures, not verified
first-publication latency; multiple timestamped snapshots and source revision
history are needed to measure variability and revisions. The report is saved
under a new `outputs/expanded/shadow_evaluation_data/issue_time_audit_*`
directory.

The November 6 and November 7 refreshes both had latest CHIRPS dated September
30 and latest ERA5-Land dated October 26; their common complete cutoff was
September 30, 37 and 38 days old at retrieval, respectively. Their overlapping
273 daily values per source and all 549 monthly rows were identical at
`1e-10` tolerance. This confirms no observed revision between these two
retrievals, not a general revision rate or true publication latency; there is
still only one distinct common-data cutoff. The archived
September-origin/October-target shadow row was created November 6, after
October began, so it is only a pipeline exercise. Forecast creation now
requires both retrieval and issue timestamps to precede the target month's
start. No new prospective one- or three-month forecasts should be logged until
a fresh, audited input snapshot supports a genuinely future target window;
longer-lead shadows must use a separately versioned candidate and log.

## Required reporting

For each lead and outer fold, report:

- sample count, onset count, and onset prevalence;
- true positives, false positives, false negatives, and true negatives;
- precision, recall, F1, and precision-recall average precision;
- Brier score and Brier skill score against the fold-specific
  training-prevalence baseline;
- probability calibration diagnostics;
- false warning episodes per year, detected onset episodes, and timing/lead;
- model, features, transformation, hyperparameters, selected threshold, and
  their training/validation provenance.

Report raw event counts alongside rates. Use event- or year-block resampling
for uncertainty where support permits, and state when the number of events is
too small for stable inference. Month-level accuracy alone is not an adequate
warning-system metric.

## Candidate acceptance and external validation

A model is a research candidate only if its locked retrospective procedure
shows useful discrimination and probability skill against the simple baselines
without an unacceptable warning burden across time periods. The available
historical record has few onset events, so no numerical alert-cost threshold is
assumed by this protocol; operational stakeholders must define acceptable
misses, false alarms, and lead time before an alert policy is designed.

Before operational use, lock the model, feature pipeline, index definition,
calibration, and warning threshold, then validate them on genuinely new
observations that were not used for development. A prospective evaluation
should archive each forecast, its data cutoff, model version, probability,
threshold, and later observed outcome. A separately evaluated catchment may
provide additional evidence but is not interchangeable with temporal
validation at Mnjoli.

Until that independent evidence exists, describe onset results as exploratory
research. Keep them separate from the live continuous-anomaly forecast and do
not send or advertise them as operational drought alerts.
