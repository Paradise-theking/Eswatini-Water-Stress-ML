# H SAF Order 53877 — Clarification Follow-up

**Catchment:** Mnjoli W60E, Eswatini  
**Order reference:** 53877  
**Products:** H64, H90, H122 and H14  
**Clarification email sent:** 12 November 2026  
**Status:** Awaiting H SAF response

## Purpose

A clarification email has been sent to the H SAF team following the
single-day pilot retrieval and preliminary inspection documented in
`HSAF_Mnjoli_Data_Retrieval_Methodology.pdf`.

## Questions requiring clarification

1. What exact start and end times does the H90 00 UTC 24-hour product
   represent?
2. How are the timestamps of the H90 hourly files defined: interval
   start or interval end?
3. Are the H64 and H90 accumulation windows directly comparable for
   the 26 October 2026 sample?
4. What product-specific quality indicators and screening rules should
   be applied when comparing these precipitation products?

## Current evidence and limitations

- All four downloaded archives passed their supplied MD5 checks.
- H64 reported 0.0 mm for its 24-hour accumulation ending
  26 October 2026 at 00 UTC.
- H90 reported a catchment-pixel mean of 7.48 mm for the file
  timestamped 26 October 2026 at 00 UTC.
- The exact temporal alignment of these two products has not yet
  been independently confirmed.
- The H90 hourly nominal-label sum was 6.01 mm; its interval-label
  convention still requires clarification.
- No independent date-matched rainfall observation was available
  for validating the discrepancy.

## Research decision pending clarification

The H64 and H90 values must remain separate. They must not be averaged,
treated as interchangeable, or used to calculate reservoir volume.

H122 soil-moisture observations and H14 soil-wetness indices must also
retain their respective spatial, temporal and measurement limitations.

No H SAF product has yet been incorporated into the operational
forecast model on the basis of this pilot.

## Next actions

1. Record and archive the H SAF response when received.
2. Confirm accumulation windows and timestamp conventions.
3. Obtain independent rainfall observations for matching dates.
4. Repeat comparisons across multiple dates and seasons.
5. Document quality screening and spatial representativeness before
   considering model integration.

**Reference:** H SAF Data Retrieval and Pilot Methodology,
Mnjoli catchment, Order 53877, 11 November 2026.
