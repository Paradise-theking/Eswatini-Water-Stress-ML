# Mnjoli Water-Stress Research Record

## Data, methods, results, deployment, and research limitations

**Version:** 1.0  
**Prepared:** 7 November 2026  
**Study focus:** Mnjoli W60E catchment, Eswatini  
**Status:** Research record for manuscript preparation; not peer reviewed

> **Scope and interpretation.** This document records several related but
> distinct research experiments and the current remote-data prototype. It is
> not a claim that the system forecasts reservoir storage, water supply, or
> drought impacts. The live dashboard currently reports a satellite-derived
> surface-area proxy and a weather-ensemble hydroclimate context as separate
> indicators. Operational storage observations sufficient for calibration
> and validation have not been obtained.

## 1. Executive summary

The project began as a machine-learning study of monthly hydroclimatic water
stress in Eswatini. It used remotely sensed precipitation and reanalysis-derived
meteorological and land-surface variables to predict a standardized water
balance one month ahead. The research was subsequently broadened to examine
drought-onset warnings and the reliability of historical rainfall inputs.
Because there is no verified, timestamped Mnjoli storage series, inflow or
release history, or stage-area-volume curve in the project data, the deployed
Mnjoli Water-Risk Watch does not estimate reservoir storage. It presents:

1. Sentinel-2 visible reservoir surface area inside an informal mapped water
   polygon; and
2. a 30-day Open-Meteo seasonal ensemble summary for precipitation, reference
   evapotranspiration (ET0), and precipitation minus ET0.

Those indicators are explicitly not interpreted as reservoir volume, inflow,
or service-risk probabilities.

The expanded historical water-balance index is reproducible over 1981-2025.
However, drought-onset events are rare (30 onsets in 540 months), and
retrospective warning experiments show a sensitivity-versus-warning-burden
trade-off. The most recent one-month warning-budget screen detected 3 of 13
onsets in pooled, previously examined development folds. This is not sufficient
evidence for an operational alert. A separate 2020-2025 continuous-anomaly
test showed better average error for a regularized model than persistence, but
contained only one month below the drought threshold; it cannot establish
drought-detection skill.

A CHIRPS v2-to-v3 overlap study found substantial historical product differences.
On the shared 1981-2025 period, v3 precipitation total was 19.23% higher than
the existing v2 catchment series. A full-period matched-reducer audit found the
v2 spatial-reducer difference was small compared with the v2/v3 difference.
Nevertheless, v3 processing, daily disaggregation, and mirror rounding remain
possible contributors, so this is not a pure causal attribution to product
version alone.

The water-risk API and Firebase-hosted dashboard are deployed. Live endpoint
checks returned weather ensemble data and a Sentinel-2 surface-area estimate.
These are operational checks of data retrieval and display, not hydrological
validation.

## 2. Research questions and project evolution

The original research question was whether environmental conditions observed
at month *t* could predict a standardized hydroclimatic water-stress index at
month *t+1*. The work emphasized chronological validation, leakage prevention,
regularization, and comparison against simple baselines.

The project then explored a second question: can a model issue useful warnings
before a month crosses a drought threshold? Continuous index regression and
binary onset classification were treated as different targets, with different
metrics and limitations.

The intended application later shifted toward water availability at Mnjoli and
a useful outlook up to 30 days ahead. Source review did not identify reliable,
dated reservoir-operation records. The product was therefore deliberately
reframed as a **relative satellite-and-weather proxy**, rather than a synthetic
storage forecast. The user-facing prototype keeps the satellite observation
and seasonal weather context separate.

These phases are not one uninterrupted model experiment. In particular:

- The original short-record Ridge test, expanded historical continuous-index
  work, onset-warning backtests, CHIRPS transition study, and live proxy use
  different targets, inputs, periods, and evaluation designs.
- Results from one phase should not be combined into a single headline
  performance estimate.
- Deployment of an API or dashboard demonstrates software integration, not
  predictive validity.

## 3. Study area and data

The focal geography is the Mnjoli W60E catchment in Eswatini. Historical
catchment products use a HydroSHEDS basin polygon. The satellite prototype
uses a separate OpenStreetMap water polygon for Mnjoli Dam. The reservoir
water polygon is an informal mapped geometry, not a surveyed maximum-pool
boundary, and it must not be used to infer storage volume without additional
hydraulic information.

| Data or product | Role | Coverage / resolution | Important qualification |
|---|---|---|---|
| CHIRPS v2 daily (`UCSB-CHG/CHIRPS/DAILY`) | Historical precipitation for the existing expanded index and live ingestion workflow | 0.05-degree grid; historical archive 1981-2025, refreshed through 2026-09-30 in the audited snapshot | Delayed and revisable; retrieved series does not by itself establish what was available at forecast issue time |
| ERA5-Land daily aggregate | PET and meteorological / land-surface predictors | Historical archive through 2025; refreshed through 2026-10-26 in the audited snapshot | Common complete cutoff with CHIRPS was 2026-09-30 at the November 2026 refresh |
| CHIRPS v3 final/rnl mirror | Version-transition sensitivity analysis | 0.05-degree grid; 1981-01 through 2025-12 | Public dynamical.org processed mirror; daily values are disaggregated and rounded; final release is typically about 2-3 weeks after month end |
| Sentinel-2 Surface Reflectance Harmonized | Recent satellite surface-water extent proxy | Scene-based imagery; 20 m processing scale for the prototype's area reductions | Cloud/shadow masking, pixel classification, shoreline effects, and mapped-boundary uncertainty affect estimates |
| Open-Meteo seasonal forecast (ECMWF EC46 context) | 30-day weather ensemble | About 36 km model-grid scale; 50 paired precipitation and ET0 members in the observed live response | Coarse spatial representation and extended-range uncertainty; not a catchment runoff or reservoir-inflow forecast |
| OpenStreetMap relation 1792362 | Mnjoli Dam water polygon for satellite processing | Polygon geometry; 908 vertices in the retrieved relation | Informal mapped water boundary; attribution to OpenStreetMap contributors required |
| EWSC public dam-level display | Investigated as a potential storage observation | Public homepage display sampled during the research | “102%” had no observation timestamp, percentage definition, or usable history; repeated archived display is not validation ground truth |

### 3.1 Historical refresh and issue-time freshness

The audited shadow-data refresh retrieved CHIRPS through 30 September 2026 and
ERA5-Land through 26 October 2026. Their common complete daily cutoff was
30 September 2026, 38 days before the 7 November refresh. The refreshed
monthly series covered 1981-01 through 2026-09 (549 months), preserved the
existing 540 historical months through 2025 within tolerance, and passed its
data audit.

Two retrieval snapshots on 6 and 7 November had the same common cutoff and
identical overlapping source values. This only describes those sampled
retrievals. It does not establish a general first-publication delay or a
revision rate. A separate October-target shadow forecast was created after
October had started and is documented as a pipeline exercise, not timely
forecast evidence.

## 4. Index construction and label definitions

For the expanded historical analysis, monthly climatic water balance is:

`WB_t = P_t - PET_t`

where `P_t` is catchment precipitation and `PET_t` is potential
evapotranspiration. The three-month balance is the rolling sum of monthly
water balance:

`WB3_t = WB_t + WB_(t-1) + WB_(t-2)`

The standardized water-balance anomaly is calculated separately for each
calendar month using the 1981-2010 reference period:

`SWBA_t = (WB3_t - mean_calendar_month,1981-2010) / sample_SD_calendar_month,1981-2010`

The reference standard deviation uses sample standard deviation (`ddof=1`).
Negative values denote drier-than-reference hydroclimatic conditions; positive
values denote wetter-than-reference conditions. SWBA is a standardized
climatic anomaly, not a probability, reservoir fullness percentage, or direct
water-supply measure.

For onset classification:

- A drought month is `SWBA <= -1.0`.
- An onset is the first drought month following an immediately preceding valid
  month with `SWBA > -1.0`.
- A forecast origin is eligible only when its origin-month SWBA is greater
  than -1.0, so the task is advance warning rather than detecting an already
  active drought.
- The primary warning horizon is one month. Two- and three-month analyses are
  exploratory.

The original short-record Ridge study used a distinct, training-period
climatology and a 2024-2025 independent test. Those values must not be merged
with expanded SWBA results using the fixed 1981-2010 reference.

## 5. Historical data and label audit

The expanded monthly dataset contains 540 consecutive months from January 1981
through December 2025. It has two missing SWBA / three-month-balance rows at
the beginning of the rolling window and no missing precipitation, PET, or
first-layer soil-moisture predictor rows. The audit reconstructed monthly
balance, rolling three-month sums, reference climatology, and SWBA from source
fields.

| Audit check | Maximum absolute difference | Outcome |
|---|---:|---|
| Monthly balance vs precipitation minus PET | 8.97e-14 | Passed |
| Three-month balance vs rolling sum | 2.27e-13 | Passed |
| Stored calendar-month reference mean | 1.14e-13 | Passed |
| Stored calendar-month sample standard deviation | 4.26e-14 | Passed |
| Stored vs reconstructed SWBA | 5.44e-15 | Passed |

The audit found 77 drought months and 30 onset events. Eligible samples with
complete predictors numbered 460, 459, and 458 for one-, two-, and three-month
leads respectively. Onset counts are small and uneven across chronological
blocks; the one-month test blocks contain 5, 5, 2, and 1 events for
2006-2010, 2011-2015, 2016-2020, and 2021-2025.

## 6. Modeling experiments and results

### 6.1 Original short-record continuous forecast

The repository documents a leakage-aware Ridge Regression pipeline with
median imputation, standardized predictors, and Ridge alpha 0.05. Its target
is a one-month-ahead continuous stress index. Feature ranking and model
configuration were selected without random temporal shuffling; the
24-observation independent test covers January 2024 through December 2025.

| Test result | Ridge |
|---|---:|
| MAE | 0.9625 |
| RMSE | 1.3906 |
| R-squared | 0.2124 |
| Test RMSE: persistence | 1.7568 |
| Test RMSE: climatology | 1.7589 |

The README reports an approximately 20.84% RMSE reduction versus persistence
and 20.93% versus climatology on that specific test. Only one test observation
was at or below SWBA -1.0, and none was at or below -1.5 or -2.0. Two extreme
absolute anomalies in this test were both positive, not drought events.
Therefore this test is not meaningful evidence of dry-event or drought-warning
skill.

### 6.2 Expanded continuous-anomaly experiment

A later expanded analysis evaluated January 2020 through December 2025 using a
longer historical series and a different fixed-reference SWBA definition.
Reported final-test metrics were:

| Model / benchmark | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Lasso | 0.5089 | 0.7001 | 0.5155 |
| Ridge | 0.5077 | 0.7123 | 0.4985 |
| Tuned XGBoost | 0.5420 | 0.7465 | 0.4491 |
| Persistence | 0.6952 | 0.9833 | 0.0443 |
| Seasonal climatology | 0.7970 | 1.1294 | -0.2608 |
| Training-mean baseline | 0.8040 | 1.1435 | -0.2924 |

Rolling-origin validation over the reported folds favored an ElasticNet/Lasso
configuration on average (mean RMSE 0.5253, mean R-squared 0.5804), with
Ridge close behind (mean RMSE 0.5304, mean R-squared 0.5710). These are
continuous-target results on the expanded SWBA and do not demonstrate
forecasting of reservoir level, storage, inflow, or service impacts. The
2020-2025 holdout contained only one SWBA drought month at or below -1.0, so
continuous average scores must not be recast as drought detection performance.

### 6.3 Drought-onset warning experiments

The onset task was audited and evaluated separately. Prior exploratory model
searches were replaced by chronology-aware screens and a warning-budget
feasibility analysis. Previously examined 2006-2025 folds are development
backtests, not an untouched final test. Thresholds were selected on a prior
five-year validation block for each outer test block, under user-selected
limits of no more than 0.5 false-warning episodes per year and 3 false-warning
months per year.

In the November 2026 pooled one-month warning-budget screen, the fixed
four-feature hydroclimate logistic model:

- detected 3 of 13 unique onset events (event recall 0.231);
- produced 7 false-warning months and 5 false-warning episodes over four
  five-year test blocks; and
- met both prespecified warning limits in all four historical folds.

The seasonal and training-prevalence baselines produced no warnings and
detected no onset in the one-month pooled test. Meeting the false-warning
limits by issuing few warnings is not sufficient if event sensitivity is
low. At two months, the hydroclimate candidate detected 8 of 13 events but
met both limits in only 2 of 4 folds (11 false-warning months and 10
episodes). At three months, it detected 11 of 13 but met both limits in 2
of 4 folds (16 false-warning months and 12 episodes). Overlapping forecast
windows make origin-level outcomes dependent; an onset is counted once for
event recall even if several warnings cover it.

A separate nested one-month backtest artifact reports a selected-model
summary of 3/13 onset hits, 12 false-warning months, and 5 false-warning
episodes across 209 test samples; its Brier score was 0.0553 versus 0.0589
for its training-prevalence baseline. This is a different experiment and
threshold-selection artifact than the fixed-method warning-budget screen.
Both are retrospective, previously examined development results and should be
reported as exploratory, not as independent confirmation.

The partial 2026 holdout covered January through September and contained zero
observed onsets. Its Brier score was 0.00054 against 0.00523 for the
prevalence baseline, but with zero events it cannot estimate onset recall or
detection skill. Zero warnings in that partial period likewise cannot
establish a long-run warning burden.

### 6.4 CHIRPS v2-to-v3 comparison

The comparison used complete catchment-month coverage for January 1981 through
December 2025 (540 months). CHIRPS v3 final/rnl daily fields were obtained
from the dynamical.org Icechunk/Zarr mirror of the CHC product; existing
ERA5-Land PET was held fixed. V3 was evaluated with the existing v2
1981-2010 reference statistics and again after recalculating the reference
statistics from v3. This separates reference-period effects from the
precipitation series change.

| Comparison | Result |
|---|---:|
| Existing v2 total precipitation | 31,836.12 mm |
| V3 total precipitation | 37,958.46 mm |
| V3 minus v2 total | +6,122.34 mm (+19.23%) |
| Monthly total correlation | 0.94694 |
| Monthly mean bias (v3 minus v2) | +11.34 mm |
| Monthly MAE / RMSE | 15.16 / 24.90 mm |
| Drought months, v2 | 77 |
| Drought months, v3 using v2 reference | 54 |
| Drought months, v3 with its own 1981-2010 reference | 75 |
| Onsets, v2 / v3 with v2 reference / v3 own reference | 30 / 25 / 32 |

With the v2 reference retained, 27 drought-month labels changed; with v3's own
reference, 24 labels changed. Those are changes to historical index labels,
not changes in model skill. The data came from a processed mirror that rounds
stored values and documents ERA5-based daily disaggregation of pentadal
CHIRPS totals. Thus the differences cannot be attributed solely to the
version-number change.

A full-period reducer-equivalence audit sampled all 11 native v2 cells
intersecting the catchment for 16,436 days (180,796 pixel-day samples), then
applied the same fractional-area weighting method used for v3. The weighted
v2 total was 31,802.34 mm versus 31,836.12 mm for the archived Earth Engine
mean reducer: -33.77 mm (-0.106%) over 45 years. Daily correlation was
0.999989, daily MAE 0.00860 mm, and maximum annual-total difference 0.218%.
Under this matched spatial reducer, v3 remained 19.36% above v2. This
indicates that the reducer difference is small relative to the observed
precipitation-product difference, while not resolving mirror processing,
daily disaggregation, or rounding as explanations.

V3 final data are not used in the live model. Its reported typical final
release delay of two to three weeks after month-end also does not by itself
meet an early-warning freshness requirement.

## 7. Deployed relative water-availability prototype

### 7.1 Satellite surface-area method

The backend queries the Google Earth Engine collection
`COPERNICUS/S2_SR_HARMONIZED` over the Mnjoli OSM water polygon and a
90-day lookback, limited to 100 images. It computes the modified normalized
difference water index using green (B3) and shortwave infrared (B11)
reflectance:

`MNDWI = (B3 - B11) / (B3 + B11)`

Pixels with MNDWI greater than zero are classified as water. Sentinel
scene-classification pixels 4, 5, and 6 are treated as valid; clouds, shadows,
snow, and unclassified pixels are excluded. Water and valid areas are
calculated using pixel area at a 20 m reduction scale. A scene is retained
only when valid pixels cover at least 80% of the mapped polygon. Multiple
images on the same acquisition date are deduplicated, keeping the observation
with better valid coverage. The service returns the two most recent
qualifying dates and their relative area change when available.

This is a spectral surface-extent proxy. It is not a water-level measurement
or storage-volume conversion. It depends on the OSM polygon, valid-pixel
coverage, classification threshold, atmospheric/surface conditions, and
shoreline-pixel behavior. No local reference labels or surveyed stage-area
curve were available to calibrate the threshold or validate area accuracy.

### 7.2 Weather-ensemble method

The service requests 30 daily precipitation and reference ET0 values from the
Open-Meteo seasonal forecast API at the Mnjoli-area requested coordinates.
Precipitation and ET0 are paired by ensemble-member identifier. For each
complete member, daily values are summed over the 30-day period, and
precipitation minus ET0 is calculated using the paired member totals. The API
returns the P10, median, P90, and arithmetic ensemble mean. Fewer than 10
complete paired members or invalid 30-day series are rejected.

Gross rainfall volume is calculated as catchment precipitation depth
multiplied by the 896.9 km2 catchment area. Since one millimetre over one
square kilometre equals one megalitre, the output unit is ML. This is gross
rainfall before interception, infiltration, runoff generation, routing,
reservoir operations, or withdrawals. `P - ET0` is a hydroclimate indicator,
not runoff, reservoir inflow, storage change, or water supply.

### 7.3 Live snapshot (retrieval: 7 November 2026)

The public API was directly checked after deployment. The following values
describe that retrieval snapshot; they are not an archived hindcast or a
verified observation series.

| Live indicator | Snapshot result |
|---|---:|
| Satellite acquisition date | 2026-10-31 |
| Estimated surface area | 12.656 km2 |
| Valid-pixel coverage | 92.46% |
| Previous qualifying scene | 2026-10-19 |
| Previous area / coverage | 13.643 km2 / 99.37% |
| Relative area change | -7.24% |
| Weather period | 2026-11-07 to 2026-12-06 |
| Weather ensemble members | 50 |
| Precipitation total, P10 / median / P90 | 54.76 / 98.65 / 155.68 mm |
| Reference ET0 total, median | 141.06 mm |
| P - ET0, P10 / median / P90 | -108.56 / -43.94 / +32.83 mm |
| Gross catchment rainfall, median | 88,479 ML |
| Resolved model grid point | -25.9953, 31.8103; elevation 308 m |

The relative surface-area change must not be phrased as a 7.24% storage
decrease. The seasonal ensemble median climate balance being negative does not
mean the reservoir will lose a corresponding volume; there is no calibrated
catchment-to-reservoir response model.

### 7.4 API and user interface

The FastAPI application exposes:

- `GET /water-risk/climate-balance`
- `GET /water-risk/surface-area`

The frontend dashboard is hosted at
<https://eswatiniwaterstress.web.app/water-risk>. On 7 November 2026, both
endpoints returned HTTP 200 and the public dashboard displayed the dated
satellite and forecast indicators. A deployment bug initially created an
Earth Engine geometry before initializing the Earth Engine client. The
initialization order was corrected and regression-tested; the endpoint
subsequently returned a valid scene summary. This incident is an
implementation-history note, not a data-quality result.

The dashboard links and response payloads describe source and retrieval time,
but a persistent time-series archive of repeated satellite and forecast
snapshots still needs to be established for retrospective verification.

## 8. Evidence-based interpretation

The evidence supports three limited conclusions:

1. A reproducible monthly hydroclimatic anomaly series can be constructed
   from the archived precipitation and PET products, and a continuous
   one-month-ahead model can outperform simple persistence/climatology
   baselines on some historical test designs.
2. The drought-onset problem is event-sparse. Retrospective results show that
   warning-budget compliance can coexist with low onset sensitivity, and
   apparent sensitivity rises at longer horizons with increased warning
   burden. The evidence does not justify operational drought alerts.
3. Remote satellite/weather services can populate a useful, clearly limited
   monitoring interface. They currently provide independent contextual
   indicators, not a validated water-availability or storage forecast.

The evidence does **not** show that the dashboard predicts water shortage,
reservoir storage, service interruption, or operator action. Nor does an API
response establish the physical accuracy of MNDWI-derived area or seasonal
forecast skill over the catchment.

## 9. Limitations and unresolved research requirements

### Hydrological and operational data

- No verified dated Mnjoli storage or level observations, inflow, release,
  downstream discharge, abstraction, demand, or service-impact time series
  were identified in the project data.
- The public 102% value lacks timestamp and definition, and sampled archives
  repeat it. It is not a calibration target.
- There is no surveyed reservoir stage-area-volume relation for converting
  image-derived area to level or storage.
- No local rain-gauge series or independent satellite-area reference labels
  were used for validation.

### Forecast and remote-sensing uncertainty

- The 30-day seasonal weather product is coarse-grid extended-range guidance,
  not a locally calibrated catchment forecast.
- P - ET0 omits soil storage, runoff generation, routing, groundwater,
  reservoir operation, and demand.
- Satellite results are sensitive to cloud masking, valid coverage, water
  threshold, shoreline mix, scene availability, and polygon quality.
- The mapped OSM water polygon is not a surveyed maximum extent.
- Live responses observed on one retrieval date do not provide temporal
  accuracy, forecast reliability, or trend significance.

### Statistical evidence

- The drought-onset series has only 30 historical events, unevenly distributed
  in time; outer test blocks contain very few events.
- Development folds have already influenced model experimentation and are not
  independent validation.
- Overlapping lead windows produce dependent samples and warnings.
- The 2026 partial holdout has no onset, so detection skill cannot be estimated.
- Input retrieval freshness and source vintages do not yet establish exact
  issue-time availability or general source-revision behavior.
- Results from the 2015-2021/2024-2025 original Ridge experiment and expanded
  1981-2025-reference experiments use different targets/reference designs
  and must be reported separately.

## 10. Recommended next research steps

1. **Archive each feed.** At fixed retrieval intervals, persist raw API
   payloads, retrieval timestamps, source version/model, issue and valid
   dates, resolved coordinates, and hashes. Keep each forecast vintage
   immutable.
2. **Obtain operator observations.** Request dated level/storage, inflow,
   releases, operating rules, capacity/datum definitions, quality flags,
   service impacts, and permission from the responsible data steward.
3. **Validate satellite area.** Compare Sentinel-derived area with
   operator-approved observations or independently interpreted imagery;
   examine threshold, coverage, season, and shoreline sensitivity.
4. **Build a physical response model only when supported.** Link rainfall,
   catchment response, reservoir inflow, evaporation, releases, and demand
   with unit-checked data. Do not equate gross rainfall with inflow.
5. **Lock target and decision use.** Operators should define what outcome
   matters, the acceptable lead time, false-warning burden, and missed-event
   cost before selecting alert thresholds.
6. **Prospective non-alerting evaluation.** Freeze forecasts and inputs before
   outcome windows begin. Accumulate enough independent events and report
   calibration, discrimination, warning episodes, lead time, and uncertainty.
7. **Keep product transitions explicit.** Do not silently replace v2 with
   v3 or retrain on relabeled history. First archive vintages, compare
   processing, and assess downstream model/label sensitivity.
8. **Separate claims.** Present continuous hydroclimatic forecasting,
   onset-warning classification, satellite monitoring, and reservoir
   storage/supply forecasting as separate research claims with their own
   validation.

## 11. Reproducibility map

Repository sources of truth and evidence used to prepare this record:

- `README.md` — original monthly Ridge study, model pipeline, 24-month test,
  and application description.
- `documentation/DROUGHT_ONSET_EVALUATION_PROTOCOL.md` — index/event
  definitions, audit gate, warning-budget methodology, CHIRPS transition,
  and shadow-evaluation limitations.
- `documentation/MNJOLI_WATER_RISK_WATCH_PLAN.md` — source inventory,
  storage-data gap, product framing, and operational recommendations.
- `scripts/create_expanded_swba.py` — expanded SWBA construction.
- `scripts/audit_drought_onset_data.py` — historical index and label
  reconstruction checks.
- `scripts/evaluate_warning_budget_feasibility.py` and
  `scripts/run_nested_drought_onset_backtest.py` — chronological onset
  evaluation workflows.
- `scripts/run_chirps_v3_transition_comparison.py` and
  `scripts/audit_chirps_v2_spatial_reducer.py` — CHIRPS product/reducer
  sensitivity analyses.
- `backend/water_availability.py` — live weather and Sentinel-2 data
  processing.
- `data/external/mnjoli_reservoir_boundary.geojson` — OSM dam-water geometry
  with attribution metadata.
- `outputs/expanded/drought_onset_model_preparation/data_audit/` — audited
  1981-2025 SWBA and onset records.
- `outputs/expanded/drought_onset_model_preparation/warning_budget_feasibility_20261107T011001Z/`
  — pooled and fold-level warning-budget outputs.
- `outputs/expanded/shadow_evaluation_data/refresh_through_2026-09/` —
  refreshed data, manifest, and partial-2026 holdout evaluation.
- `outputs/expanded/chirps_v3_transition/run_through_2025_20261106T181031Z/`
  — v2/v3 comparison and full-period reducer audit.
- `outputs/expanded/final_evaluation/final_test_metrics.csv` — expanded
  continuous-anomaly test metrics.
- `outputs/expanded/rolling_origin_validation/rolling_origin_validation_summary.csv`
  — continuous-target rolling-origin results.

Generated CSVs and manifests should be preserved with their recorded input
hashes. When reporting any result, state the target/index definition, data
product and vintage, period, split design, sample/event counts, and whether
the evaluation was used during model development.

## 12. Candidate source citations

These are data-product and service references, not a complete scholarly
bibliography. Check current product documentation, access dates, licenses,
versions, and citation requirements before manuscript submission.

1. Funk, C. et al. (2026). CHIRPS v3.0. *Scientific Data*, 13, 718.
   <https://doi.org/10.15780/G2JQ0P>
2. UCSB Climate Hazards Center. CHIRPS data archive and documentation.
   <https://data.chc.ucsb.edu/products/CHIRPS/>
3. ECMWF. ERA5-Land hourly data documentation.
   <https://www.ecmwf.int/en/era5-land>
4. Google Earth Engine Data Catalog. COPERNICUS/S2_SR_HARMONIZED.
   <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>
5. Open-Meteo. Seasonal forecast API documentation.
   <https://open-meteo.com/en/docs/seasonal-forecast-api>
6. OpenStreetMap contributors. Mnjoli Dam relation 1792362.
   <https://www.openstreetmap.org/relation/1792362>
7. dynamical.org. UCSB-CHC CHIRPS Analysis Final mirror, snapshot
   `T5RTYW439H0E8VF37CQG`; processed dataset license stated as CC BY 4.0.
   <https://dynamical.org/catalog/ucsb-chc-chirps-analysis-final/>

---

**Suggested manuscript framing:** “A data-constrained hydroclimatic
monitoring and forecasting study for Mnjoli, Eswatini: index evaluation,
input-product sensitivity, and a satellite/weather proxy prototype.” Avoid
“validated reservoir storage forecast” or “operational drought-warning model”
unless and until the corresponding observations and prospective evaluation
exist.
