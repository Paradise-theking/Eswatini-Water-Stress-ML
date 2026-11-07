# Mnjoli Water-Risk Watch: Data and Decision Plan

**Status:** Relative satellite/weather proxy prototype implemented; no storage validation or operational thresholds approved  
**Prepared:** 2026-11-07  
**Scope:** Mnjoli W60E catchment and water-supply risk monitoring

## Recommendation

Reframe the product from a single monthly drought-onset classifier into a
layered **Mnjoli Water-Risk Watch**:

1. **Observed conditions:** report the newest verified rainfall, catchment
   water-balance indicators, and (if obtained) reservoir level/storage and
   inflow.
2. **Near-term outlook:** estimate the chance of worsening conditions over a
   short, explicitly stated horizon using low-latency rainfall estimates and
   weather forecasts, with uncertainty and issue-time provenance.
3. **Operational context:** translate observations into a watch or escalation
   only after the responsible water operator defines what the indicators mean
   for service risk and approves thresholds.
4. **Confirmation:** use quality-controlled CHIRPS/ERA5-Land histories for
   retrospective context and later re-analysis, not as if delayed data were
   available at forecast issue time.

The existing drought-onset and continuous SWBA products remain research
signals. Neither should be presented as a reservoir, water-supply, or service
continuity forecast.

## Prototype implementation status

The experimental **Mnjoli Water-Risk Watch** screen is available at `/water-risk`.
It presents two dated remote-data indicators separately:

- A Sentinel-2 surface-water area estimate over the OpenStreetMap Mnjoli
  reservoir polygon, with valid-pixel coverage and change from the previous
  sufficiently clear scene. The polygon is not a surveyed maximum-pool
  boundary; the result estimates visible surface extent, not level or storage.
- A 30-day Open-Meteo seasonal ensemble summary for precipitation, reference
  ET₀, and paired precipitation-minus-ET₀. Gross rainfall volume is calculated
  over the mapped 896.9 km² catchment, before runoff, infiltration, routing, or
  reservoir operations. The 30-day forecast is at extended-range lead and
  coarse model-grid scale.

The indicators are **not combined into a storage estimate**. Precipitation
minus ET₀ is not runoff or reservoir inflow, and neither input supports a
water-supply forecast. The screen explicitly notes that no dated storage
series or stage–area–volume curve was found. EWSC's 102% public display has no
measurement timestamp or percentage definition and was unchanged across
sampled archived captures; it is not treated as validation ground truth.
Consequently, no storage estimate is calibrated or scored. No operational
thresholds or alerts are applied. The existing forecast and services views are
unchanged.

The API exposes `/water-risk/climate-balance` and
`/water-risk/surface-area`. The former summarizes paired precipitation and
ET₀ ensemble members and requires at least ten complete members. The latter
filters Sentinel-2 scenes by valid-pixel coverage (at least 80%), deduplicates
acquisitions by date, and returns at most the two most recent qualifying
observations. Earth Engine authentication and project access are required for
the satellite endpoint; live provider calls are not covered by local unit tests.

## Repository evidence

The current repository contains historical and refreshed CHIRPS precipitation
and ERA5-Land meteorological/land-surface variables, plus derived monthly
water-balance and SWBA series. The live ingestion path uses CHIRPS and
ERA5-Land in `backend/data_ingestion.py`; `backend/live_forecast.py` forecasts
the Water Stress Index from those features.

There are no repository data files for Mnjoli reservoir telemetry, inflow or
release, river-gauge discharge, local rain-gauge observations, abstraction,
water demand, service interruptions, or restriction decisions. The project
README also identifies reservoir storage and operational water scarcity as
outside the current model's coverage.

The current live inputs are not timely enough for an early monthly warning:
the November 7, 2026 Earth Engine refresh had CHIRPS through September 30 and
ERA5-Land through October 26, leaving the common complete cutoff at September
30 (38 days old at retrieval). A November 6 snapshot had the same cutoff.
Their 273 overlapping daily values per source and all 549 monthly rows were
identical, but two snapshots with one common cutoff cannot establish a typical
publication delay or a general revision rate. See the
[issue-time audit](../outputs/expanded/shadow_evaluation_data/issue_time_audit_20261107T012318Z/issue_time_audit.json).

## Candidate data sources and evidence status

| Signal | Current evidence | Potential use | Status / next verification |
|---|---|---|---|
| **Mnjoli dam level / storage** | The [EWSC public website](https://ewsc.co.sz/) exposes a “Water Dam Levels” section and displays a Mnjoli percentage. The display is not accompanied by an observation timestamp, a definition of the percentage (including the meaning of values above 100%), a history, or a documented API in the page view checked on 2026-11-07. | Direct state-of-supply indicator, if the operator confirms the measurement, datum/capacity, timestamp, update cadence, and relevant rule curves. | **Highest-value data access request.** Ask EWSC and the responsible dam operator to confirm ownership and request daily or higher-frequency historical level/storage, inflow, release/outflow, missing-data flags, units, timestamps, quality flags, operational thresholds, and permission to use them. Do not scrape the public display as a production feed. |
| **Inflow / release / downstream flow** | Not found in the repository; no verified public time series located in this bounded check. | Distinguishes low supply from reservoir management or release effects; supports persistence and lead-time assessment. | Request a dated export and metadata from the operator. Treat ownership and access as unconfirmed until the data steward responds. |
| **Local rain gauges** | No local gauge series in repository. | Ground-truth for catchment rainfall; can correct satellite bias and identify local storm conditions. | Ask the national meteorological service and relevant water/agriculture agencies about station location, record length, reporting latency, quality control, and data-sharing rights. |
| **NASA GPM IMERG Early precipitation** | NASA describes IMERG Early as low latency (nominally about four hours), with half-hourly, daily, and other products. Early is less processed than Late/Final; it uses forward propagation. NASA states Final is intended for research and has much longer latency. [NASA IMERG product and latency notes](https://gpm.nasa.gov/data/imerg). | Experimental daily/weekly rainfall monitoring while CHIRPS is unavailable; potentially useful for a timely rain deficit or intense-rain watch. | Candidate, not a replacement for CHIRPS by assumption. Verify product/version, access/authentication, availability over Mnjoli, grid-to-catchment aggregation, bias against gauges/CHIRPS, revisions, and actual retrieval timestamps. Keep Early and Final series separately versioned. |
| **Short-range weather forecast** | Open-Meteo documents precipitation and ensemble-based precipitation probability; its documentation describes a 30-member forecast ensemble at roughly 0.25-degree resolution for that probability field. [Open-Meteo forecast documentation](https://open-meteo.com/en/docs). | A short-lead outlook for forecast rainfall and evaporative demand, not a 1–3 month seasonal outlook. | Experimental only. Record provider/model/issue time/valid time and compare archived forecasts with later observations. Confirm terms, geographic representativeness, forecast revision behavior, and whether a locally endorsed forecast source is preferred. |
| **CHIRPS / ERA5-Land quality-controlled history** | In project; useful for long-term context. Recent common cutoff lagged retrieval by 38 days. | Historical drought characterization, calibration/reference climatology, post-event verification. | Keep as a slow confirmation stream. Never backfill final values into an archived real-time forecast without retaining the original vintage. |
| **Impacts and actions** | No verified service-interruption, restriction, demand, abstraction, or impact labels found in the repository. | Defines what “useful warning” means and lets the system evaluate operational value rather than only SWBA threshold crossing. | Ask water managers which decision they need to make, how early, and which observed impact is available for verification. |

### Source notes

- The EWSC page establishes that a public dam-level display exists, not that it
  is a validated live telemetry feed or a historical dataset. Its current
  displayed percentages should not be interpreted without the operator's
  definition and timestamp.
- NASA IMERG Early's stated latency makes it a plausible fast rainfall
  candidate, but a rainfall estimate alone does not provide reservoir state or
  water-supply risk. It must be evaluated locally before operational use.
- A forecast API's existence does not imply forecast skill for Mnjoli. Forecast
  vintages must be archived before verification; retrospective reforecasts
  cannot substitute for issue-time forecasts.

## Decision framework to validate with operators

Do not invent numeric action thresholds. First agree the action and measure
with the authority responsible for it.

| Product tier | What it may say | Evidence needed before release |
|---|---|---|
| **Data status** | Latest source timestamp, cutoff, missingness, and whether observations are preliminary or revised. | Automated freshness/coverage checks; clear source and vintage. |
| **Hydroclimate watch** | Recent rainfall deficit/surplus, water-balance anomaly, and near-term rainfall outlook. | Archived low-latency inputs; local validation; uncertainty and calibration. |
| **Reservoir status** | Verified storage/level trend and inflow/release context. | Operator-provided measurements, datum/capacity definitions, update cadence, and quality flags. |
| **Operational watch / escalation** | A risk statement tied to an agreed management decision. | Operator-approved indicator, threshold or rule curve, acceptable false-alarm/miss trade-off, lead time, escalation owner, and communication procedure. |

The service should explicitly show **data freshness** and **confidence** alongside
every risk statement. If critical inputs are stale or absent, downgrade to
“data unavailable / no current assessment”; do not silently substitute a
research estimate or issue a reassuring forecast.

## Proposed implementation sequence

### Phase 1 — Stakeholder and data-access discovery

1. Identify the authority responsible for Mnjoli operations and the intended
   decision users.
2. Request historical reservoir level/storage, inflow and release data with
   timestamps, units, definitions, quality flags, revision history, and
   operational rule curves.
3. Request local rain-gauge series and any relevant water restrictions, supply
   interruptions, abstraction, or demand indicators.
4. Agree a concrete decision, lead time, action owner, and consequences of
   misses versus false alarms.

**Exit gate:** at least one dated operational outcome/measurement is available
with sufficient history and a named data steward, or the project explicitly
continues as a hydroclimate watch rather than a supply-risk warning.

#### Data request checklist

Request the longest available historical record, preferably daily or finer,
and ask the data steward to include:

- observation timestamp, timezone, publication/export timestamp, and units;
- reservoir level and/or storage volume, reference datum, usable capacity,
  and the documented interpretation of percentages above nominal capacity;
- inflow, controlled release/outflow, spill, and downstream flow where
  measured;
- station/sensor location, sampling interval, quality flags, missing-value
  codes, calibration or sensor-change history, and known gaps;
- original values plus any corrected/revised values and their revision dates;
- operator-defined rule curves, operating zones, restriction history, and
  the decisions/actions associated with those states;
- data ownership, access permission, reuse/redistribution terms, and the
  appropriate contact for technical questions.

Suggested flat-file schema (one row per timestamp and measured variable):

```text
timestamp_utc,site_id,variable,value,unit,quality_flag,published_at_utc,revision_id,source_file
```

Ask decision users these questions before choosing a model target:

1. What decision would a warning change, and who is authorized to act?
2. What is the minimum useful lead time (days/weeks/months)?
3. Which operational condition matters: low storage, falling inflow,
   restrictions, irrigation shortfall, or another impact?
4. Which events are costly false alarms versus costly misses?
5. What existing operating rules or drought plans already govern action?
6. How often can an update be acted upon, and what source delay is acceptable?

### Phase 2 — Parallel source pilot

1. Keep CHIRPS/ERA5-Land as the slow reference.
2. In a separate, versioned dataset, archive GPM IMERG Early and a short-range
   forecast at fixed issue times. Store source/product version, retrieval and
   issue timestamps, valid dates, spatial extraction method, raw-file hash,
   and any later replacement/revision as a new vintage.
3. Compare rainfall totals and event timing to local gauges, then to the
   quality-controlled reference products. Quantify missingness, bias,
   extremes, and actual end-to-end delay for the catchment.
4. Do not splice a fast product into the existing model inputs or overwrite
   final/revised observations.

**Exit gate:** repeatable ingestion, known source terms/access, demonstrated
issue-time availability, and acceptable local measurement behavior.

### Phase 3 — Operational target and forecast evaluation

1. Define outcomes tied to operations (for example, operator-defined low
   storage, inflow deficit, or a service restriction), separately from the
   existing SWBA drought label.
2. Establish simple benchmarks first: current reservoir status/persistence,
   seasonal climatology, and forecast-informed hydrologic balance where inputs
   support it.
3. Evaluate chronologically and prospectively. Freeze each issue-time forecast
   and input vintage before the outcome window begins. Report lead time, event
   detection, false-warning episodes/months, uncertainty, and operational
   impact—not month-level accuracy alone.
4. Start non-alerting shadow operation only for targets that have not begun at
   issue time. Keep every candidate's log separate and immutable.

**Exit gate:** decision-makers accept the target/threshold and prospective
evidence supports usefulness at the actual issue time and warning burden.

## Immediate decision

The highest-value next task is a **data access and decision discovery package**,
not another classifier. The project can prepare the request and a one-page
operator interview checklist, but it cannot obtain private telemetry without
the data steward's cooperation. In parallel, a small IMERG Early rainfall
pilot is technically plausible, but it should begin only after confirming the
NASA access method and should remain a non-alerting experiment.
