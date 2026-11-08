# Mnjoli Water-Availability Study
## Operator Data Request and Technical Questionnaire

**Prepared:** 7 November 2026  
**Project:** Eswatini Water Stress ML — Mnjoli water-availability research prototype  
**Purpose:** Identify fit-for-purpose observations, define an operationally meaningful target, and agree a safe route for data access.

> **Important:** Address this pack to the organization or organizations confirmed to own, operate, or hold the relevant Mnjoli records. It does not assume that a particular agency operates the dam or holds each dataset. Answer “unknown,” “not held,” or “not shareable” where appropriate. No control-system access, credentials, security-sensitive operating details, or personal customer data are requested.

## 1. Suggested cover message

**Subject:** Request for technical discussion and historical data to assess Mnjoli water-availability forecasting

Dear Mnjoli Dam / water-resource operations representative,

I am conducting a research project on whether environmental observations and remote-sensing data can support a useful, clearly qualified outlook for water availability associated with Mnjoli. The current prototype displays satellite-derived reservoir surface area and a 30-day weather-ensemble context as separate indicators. It does not estimate reservoir storage or issue operational advice.

I am seeking an initial technical discussion to understand the responsible organizations, available records, data definitions, and the decisions that would make an outlook useful. If the data steward and organization approve, I would also like to request a historical, quality-flagged export of relevant observations and the metadata needed to interpret them. The requested data would be used to assess feasibility, validate remote indicators, and—only if the evidence supports it—develop and evaluate a research forecast.

The attached questionnaire is intended to make the discussion efficient. It is completely acceptable to identify records that do not exist, are held elsewhere, or cannot be shared. Please indicate any confidentiality, attribution, retention, review, or publication conditions before transfer. A small sample with sensitive fields removed would be helpful for checking format before a full export.

Could you please direct this request to the appropriate data owner or technical contact, and advise whether a short meeting would be suitable?

Kind regards,  
**Name:** ____________________________________  
**Institution / affiliation:** _______________________  
**Email / phone:** ______________________________  
**Date:** _____________________________________

## 2. What the study is trying to establish

The research is **not** to build another weather forecast. The goal is to determine whether dated evidence can support a credible estimate or outlook of **Mnjoli-related water availability**, with uncertainty and clear limits.

Before any model is developed, the study needs to establish:

1. What “availability” means for the relevant decision (for example, measured reservoir level, usable storage, a defined supply margin, or an operator-defined threshold).
2. Which authority owns, measures, quality-controls, and may authorize use of the required records.
3. Whether observations have reliable timestamps, units, reference datums, and quality flags.
4. Whether remote satellite surface-area observations can be checked against contemporaneous ground observations.
5. Whether a future forecast can be evaluated against observations that were not available when the forecast was issued.

The present dashboard is a prototype. Satellite water area is not storage; rainfall minus reference ET₀ is not runoff or reservoir inflow. These signals must not be presented as storage or supply forecasts without appropriate calibration and independent validation.

## 3. Prioritized data request

Please indicate for each dataset whether it exists, who holds it, available dates, time resolution, access conditions, and whether a sample export can be shared.

### Priority A — minimum viable evidence

| Dataset | Preferred fields | Why it is needed |
|---|---|---|
| Reservoir level / stage observations | Observation timestamp; measured level; unit; vertical datum / benchmark; station or sensor identifier; quality/status flag; method or instrument changes | Direct, dated state observation; validates whether remote area changes correspond to level changes |
| Stage–storage / elevation–area–capacity relation | Elevation or stage; surface area; storage volume; units; vertical datum; effective date; source/survey/version; dead/live storage definition if used | Converts level or area into interpretable storage only where an approved curve exists |
| Operational meaning of “available water” | Approved target variable; thresholds or rule-curve bands; capacity basis; restrictions and intended decision horizon | Defines what the forecast is supposed to estimate and what outcome can be verified |
| Record and data dictionary | Field definitions, units, timezone, sampling frequency, valid ranges, codes, known gaps/revisions, ownership and permission | Prevents incorrect interpretation and enables reproducibility |

### Priority B — needed to explain and forecast changes

| Dataset | Preferred fields | Why it is needed |
|---|---|---|
| Reservoir inflow / source flows | Timestamp; discharge or volume; unit; gauge/source ID; measurement or estimation method; quality flag | Separates catchment response from reservoir operating effects |
| Releases, spill, and transfers | Timestamp or interval; flow/volume and units; destination/category; measurement/estimate flag; quality status | Explains storage changes and supports a mass balance |
| Abstraction / treated output / demand | Timestamp or interval; volume and units; aggregate by plant/sector where shareable; measured vs estimated; quality flag | Defines water leaving the system and supports supply outlooks |
| Local rainfall and evaporation observations | Station coordinates/elevation (or approved generalized location); timestamp; units; instrument; quality flags | Ground validation and local context for satellite/reanalysis data |
| Water-service impacts and management actions | Date/time or period; agreed event category; area/service zone at safe aggregation; start/end; action/outcome; reporting source | Allows validation against the decision or impact that matters, if available and shareable |

### Priority C — useful context, subject to availability

- Reservoir operating rules or rule curves, summarized to the level approved for
  research; seasonal target bands and definitions of usable storage.
- Catchment or river-gauge flow observations, including station metadata and
  gaps.
- Water-quality or environmental constraints that alter usable volume, if
  those can be shared as aggregated indicators.
- Planned maintenance, sensor outages, rating-curve revisions, structural
  changes, or other documented changes that affect interpretation.
- Historical forecast records, operator outlooks, or decision logs, if these
  have already been approved for sharing.

Do not send control-system diagrams, credentials, exact security procedures,
personal customer records, or other sensitive material through ordinary email.
If some information is restricted, ask whether a safe aggregate, redacted
extract, supervised review, or non-disclosive metadata summary is possible.

## 4. How data should be prepared

### 4.1 Preferred delivery

1. **CSV UTF-8** for tabular time series; one row per timestamp per site and
   measurement, or a clearly documented wide table.
2. **XLSX** is acceptable if sheets, columns, formulas, merged headers, and
   hidden rows/columns are documented. A CSV export is preferred for analysis.
3. **GeoJSON, shapefile, or GeoPackage** for approved spatial boundaries,
   accompanied by the coordinate reference system (CRS) and source/date.
4. Include the original export and data dictionary. Do not silently overwrite
   or “clean” the only copy.
5. Use an organization-approved secure transfer method. Confirm permissions,
   retention, publication and redistribution conditions before transfer.

### 4.2 Time-series fields

Recommended minimum columns:

| Column | Example | Requirement |
|---|---|---|
| `timestamp` | `2024-01-31T23:00:00Z` | ISO 8601; include timezone or explicitly document that timestamps are local time |
| `site_id` | `MNJ_LEVEL_01` | Stable, non-personal identifier; map to a metadata table |
| `variable` | `reservoir_level` | Unambiguous variable name |
| `value` | `684.32` | Numeric observation; do not encode missingness as zero |
| `unit` | `m` | Unit on every variable or unambiguously specified in a data dictionary |
| `quality_flag` | `approved` | Preserve original flag and code definitions |
| `method_flag` | `measured` | Distinguish measured, derived, modelled, estimated, and manually entered values |
| `source_version` | `rating_curve_2019_v2` | Instrument, processing, rating-curve, or dataset version where applicable |

For interval volumes and flows, include interval start and end, or document the
timestamp convention (for example, interval-ending). For cumulative counters,
identify resets and corrections. Do not infer a daily observation from a
monthly total without retaining the original aggregation period.

### 4.3 Required metadata for every series

- Dataset name, purpose, responsible data steward, owner, and contact role.
- Site/station ID and location information at an approved precision.
- Variable definitions, units, measurement basis, and valid range.
- Timestamp timezone, daylight-saving convention if any, interval convention,
  nominal sampling frequency, and aggregation method.
- Instrument/sensor type, installation or relocation dates, calibration and
  maintenance history, and known changes.
- Quality-control procedure, all flag values, missing-value codes, suspected
  outliers, corrections, and whether values are provisional or final.
- Vertical reference datum and benchmark for elevation/level, including any
  datum conversion history.
- Definition and effective date of capacity, live/dead storage, stage-area
  curve, rule curve, or “percent full.”
- Geographic CRS and units for coordinates or spatial data.
- Retrieval/export date, record period, revision policy, and dataset version.
- Permission, citation/attribution, privacy, security, retention, and
  publication restrictions.

### 4.4 Record length and resolution

Please provide the **longest quality-controlled period available**, even if
there are gaps or the resolution varies over time. Do not discard lower-quality
or older periods; label them. For a first feasibility check, a small,
representative sample spanning wet, dry, and ordinary conditions is useful,
followed—if authorized—by the complete approved record.

Hourly or daily level/flow records are preferable where already collected.
If only daily, weekly, or monthly aggregates exist, provide those at the
original resolution and explain how they were produced. No new monitoring
frequency is being prescribed by this request; any additional collection
should be agreed with the data owner and operator.

### 4.5 Example wide CSV header

```text
timestamp,site_id,reservoir_level_m,level_quality_flag,storage_ML,storage_quality_flag,inflow_m3_s,inflow_quality_flag,release_m3_s,release_quality_flag,abstraction_ML,abstraction_quality_flag,rainfall_mm,rainfall_quality_flag,record_status,source_version
```

The line above is illustrative only. Do not combine quantities with different
time intervals unless the interval basis is explicit. Long format is also
acceptable:

```text
timestamp,site_id,variable,value,unit,quality_flag,method_flag,source_version
```

Example:

```text
2024-01-31T23:00:00Z,MNJ_LEVEL_01,reservoir_level,684.32,m,approved,measured,sensor_A_v3
```

Use blank/NA plus an explicit quality/status field for missing or invalid
values. Preserve original precision; do not round the only copy.

## 5. Technical questionnaire

### A. Organization, ownership, and authorization

1. Which organization owns, operates, regulates, or maintains Mnjoli Dam and
   associated water-supply infrastructure?
2. Which organization holds reservoir monitoring, inflow, release, abstraction,
   and water-service records?
3. Who is the appropriate technical data steward and who can authorize
   research access or publication?
4. Are different datasets controlled by different organizations? If so, may
   you identify the appropriate contact for each?
5. Are there existing data-sharing agreements, research permits, ethics
   approvals, or internal review steps that apply?
6. Are there data classification, confidentiality, public communication, or
   security restrictions we should observe?

**Notes / referrals:** __________________________________________________

### B. Decision need and definition of availability

7. What practical decision, planning activity, or monitoring task would a
   water-availability outlook support?
8. Who would use the information, and what action could they reasonably take?
9. What does “water availability” mean in this context: physical reservoir
   storage, usable storage, deliverable supply, days of supply, a level band,
   or another operator-defined quantity?
10. Which storage/level or supply outcome would be most important to predict?
11. Is there an existing operational threshold or rule curve? Who approved it,
    when, and what exactly does each band mean?
12. How are “capacity,” “full,” “percent full,” “usable,” and any values above
    100% defined? Which datum and capacity version are used?
13. What forecast lead time would be useful (for example, days or weeks), and
    how often would information need updating to affect a decision?
14. What would be more harmful: a missed low-availability event, a false
    warning, or an imprecise estimate? Are there existing tolerances or
    escalation processes?
15. Which outcomes should a research study explicitly avoid predicting or
    advising on?

### C. Reservoir level and storage observations

16. Is a dated reservoir level/stage time series available? What is the
    earliest date, latest date, and native sampling interval?
17. Are readings instantaneous, interval means, daily extrema, manual gauge
    readings, or another statistic?
18. What are the units and vertical datum? What benchmark is used, and have
    datum, staff gauge, sensor, or reference elevations changed?
19. Is direct storage volume measured, calculated from level, or reported as
    a percentage? If calculated, what curve, capacity definition, or formula
    is used?
20. Is an elevation-stage-area-storage/capacity table available? Please
    identify its survey date, effective date, version, units, datum, source,
    and whether it distinguishes live, dead, flood, or otherwise unavailable
    storage.
21. Does the table cover current sedimentation or structural changes? When was
    it last reviewed or surveyed?
22. What instruments or procedures produce level observations? Are there
    maintenance, calibration, outage, relocation, or replacement records?
23. What quality flags, suspect-value rules, manual corrections, or
    provisional/final statuses apply?
24. Can level observations near Sentinel-2 acquisition dates be identified
    for research comparison, subject to approved data resolution and access?

**Available period / frequency / format:** _______________________________

### D. Inflow, release, transfers, and water balance

25. Are inflows to the reservoir measured or estimated? Which rivers, channels,
    or sources contribute, and where are they measured?
26. What are the units, interval conventions, rating curves, measurement
    methods, and quality flags for each inflow series?
27. Are rating curves available with effective dates and revision histories?
28. Are controlled releases, spill, transfers, or bypass flows recorded?
    Which components are measured versus estimated?
29. Are release records timestamped at the source, outlet, or destination?
    What is the interval basis and unit?
30. Are there known unmeasured exchanges, seepage, evaporation estimates,
    inter-basin transfers, or other material terms in the reservoir balance?
31. Are planned releases or operating schedules available historically, or
    only after the fact? Can a research dataset provide an approved aggregate
    without security-sensitive operating details?
32. Have dam works, outlet changes, river diversions, or operating-rule
    changes occurred during the available record?

**Available components and limitations:** ________________________________

### E. Demand, abstraction, and service outcomes

33. Are abstraction, treatment-plant output, bulk supply, or demand records
    available? What quantities, units, and time intervals are used?
34. Can these records be shared at an aggregated system or sector level
    without personal or customer-identifying information?
35. Are there historical restrictions, rationing, service interruptions,
    pressure reductions, or emergency actions that could be used as outcomes?
36. If impacts are recorded, how are events defined, dated, verified, and
    assigned to a service area or population?
37. Are impact records complete, or are there changes in reporting, coverage,
    or definitions over time?
38. What operational constraints (for example, treatment, environmental
    release, water quality, or infrastructure limits) affect the difference
    between physical storage and actually available supply?

### F. Rainfall, evaporation, and catchment observations

39. Are local rain gauges or evaporation measurements available in or near the
    catchment? Who owns them?
40. For each station, what are the coordinates/elevation at a shareable
    precision, instrument, variable, units, time basis, record period, and
    nominal sampling interval?
41. Are gauge maintenance, relocation, calibration, and quality-control
    records available?
42. Are catchment river-gauge or runoff observations available? What are the
    measurement methods, rating curves, units, and quality flags?
43. Are operational forecasts or local weather observations currently used
    by the water manager? Which products, issue times, and decision processes
    are relevant?
44. Are there known seasonal catchment, land-use, or infrastructure changes
    that influence runoff response?

### G. Data quality, history, and access

45. What is the longest period available for each requested variable, and
    where are the major gaps?
46. What are the native sampling interval and any aggregation, resampling, or
    interpolation applied before export?
47. How are missing, suspect, estimated, corrected, and provisional values
    represented?
48. Are past values revised? If so, are original vintages, revision dates, or
    change logs retained?
49. Can the data owner provide a data dictionary and a small sample export
    before the full record?
50. What file formats, delivery channels, and approval steps are preferred?
51. Is there an API or scheduled feed? If so, is it documented and approved
    for research use, and what are its access, rate, retention, and versioning
    conditions? This request does not seek credentials or control-system
    access.
52. Who should be contacted about data interpretation, quality flags, and
    corrections after delivery?

### H. Research use, publication, and collaboration

53. What uses are permitted: internal analysis, thesis/dissertation,
    conference presentation, journal publication, or public dashboard?
54. Must the organization review manuscripts, figures, quotations, or derived
    data before publication? What review period is reasonable?
55. What attribution, citation, acknowledgement, or disclaimer is required?
56. May derived statistics, model results, or anonymized/aggregated values be
    published if raw records cannot be redistributed?
57. Are there restrictions on combining these records with satellite or
    weather data?
58. What retention and deletion requirements apply when the project ends?
59. Would a technical collaborator be available to review variable meanings,
    define evaluation outcomes, and interpret results?
60. Is there an existing research ethics, institutional, or data-protection
    process that must be completed before access?

## 6. Proposed data inventory response table

Please complete one row per dataset. Add rows as needed.

| Dataset / variable | Exists? / period | Frequency / units / datum | Owner / quality flags | Shareable? / sample? |
|---|---|---|---|---|
| Reservoir level |  |  |  |  |
| Storage / percentage full |  |  |  |  |
| Stage-area-storage curve |  |  |  |  |
| Inflow / river gauge |  |  |  |  |
| Releases / spill / transfers |  |  |  |  |
| Abstraction / plant output / demand |  |  |  |  |
| Local rainfall / evaporation |  |  |  |  |
| Restrictions / interruptions / impacts |  |  |  |  |
| Other: __________________ |  |  |  |  |

## 7. Suggested minimum first delivery

If sharing a full archive requires additional approvals, an initial feasibility
package could consist of:

1. The data inventory table and data dictionary.
2. A small, redacted sample containing several representative wet/dry periods
   with original timestamps, units, quality flags, and metadata.
3. The current stage-storage/area-capacity reference and its datum/version,
   if shareable.
4. A brief explanation of how “available water” is defined and what decision
   horizon matters.
5. The process and conditions for requesting an approved historical export.

The minimum scientifically useful ground-truth set is not simply a single
“percent full” value. It requires a dated, defined level or storage series and
its reference curve/metadata. Inflow and release records are strongly
preferred to explain storage change. A separate service-impact target is
required if the intended claim is about water supply to users rather than
reservoir state.

## 8. Research safeguards

- No model or alert threshold will be tuned against confidential operational
  observations and then reported as independently validated on the same
  period.
- Data vintages, corrections, transformations, and exclusions will be recorded
  so that later results can be reproduced.
- Forecast inputs must be archived with issue/retrieval time before the target
  outcome occurs; final revised observations must not be substituted as if
  they were available at forecast time.
- Model outputs will remain research-only and non-alerting unless the data
  owner and relevant operator agree otherwise and independent evaluation
  supports the intended claim.
- Any requested aggregation, redaction, review, attribution, retention, or
  publication restrictions will be agreed with the data steward before
  analysis or release.

## 9. Contact and meeting record

**Responding organization:** ___________________________________________

**Respondent role (no personal details required in shared research copy):**
______________________________________________________________________

**Date:** ___________________  **Preferred follow-up:** _________________

**Data steward referral / next step:** __________________________________

**Conditions or restrictions to confirm in writing:**
______________________________________________________________________
______________________________________________________________________

**Unanswered questions / records to investigate:**
______________________________________________________________________
______________________________________________________________________
