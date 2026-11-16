# WaterWatch prototype-readiness data gates

## Scope

This branch hardens the existing `main` one-month-ahead Ridge inference path. It does not replace the model, change its feature specification, integrate HSAF/IMERG, or promote the expanded three-month Lasso/Ridge experiments into the live service.

The endpoint remains a research forecast. Its SWBA categories are not operator-approved impact thresholds and must not be presented as authorised water-supply or drought warnings.

## Fail-closed checks added

Before the existing live Ridge model is evaluated, the service now:

1. Reads the latest reported CHIRPS and ERA5-Land image dates from Earth Engine.
2. Selects the latest common completed calendar month.
3. Retrieves the four-month daily feature window required by the existing feature engineering.
4. Requires each source to contain exactly one observation for every expected calendar day in that window.
5. Requires the model's source fields to exist and contain finite numeric values.
6. Requires four consecutive monthly aggregates with no missing values.
7. Rejects a common observation cutoff older than `WATERWATCH_MAX_INPUT_AGE_DAYS` (default: 45 days).
8. Rejects an issue if the forecast target month has already started.
9. Includes source collection identifiers, source image dates, issue timestamp, input cutoff, feature list, model artifact SHA-256 and input fingerprint in the response.
10. Reuses an in-memory cached result only when the full monthly input fingerprint and model artifact hash are identical.

Readiness failures return HTTP 503 with an explicit unavailable reason. Other upstream inference failures return HTTP 502. The API does not fall back to a stale forecast.

## Important limitations

- Earth Engine's latest image date is not a provider publication timestamp. The current adapter does not establish when a given image first became accessible or whether it was subsequently revised.
- The input cutoff is the last day of the selected complete calendar month; it is not a claim that the data were published on that day.
- The 45-day freshness limit is a conservative engineering guard for this prototype, not a locally validated operational service-level threshold. It should be configurable and reviewed against actual observed publication latency.
- Exact daily date coverage and finite-value checks do not establish measurement accuracy, spatial representativeness, or source-vintage immutability.
- The current process-local cache and API response are not a durable immutable forecast archive. A persistent archive needs an agreed storage backend, retention policy and access controls.
- The existing model is still the original one-month Ridge artifact on `main`; this work does not make the expanded three-month regression models the deployed model.
- Existing category boundaries remain research communication categories. No official alert is authorised.

## Verification

Unit tests in `tests/test_data_quality_gates.py` cover complete coverage, missing dates, duplicate dates, missing values and missing required fields.

Run from the repository root:

```powershell
python -m unittest discover -s tests -v
```

An authenticated integration check still needs to be run in the project environment to verify the live Earth Engine responses, the real catchment and actual source latency. Do not infer successful live ingestion from unit tests.

## Next steps

1. Run unit tests and Python compilation in CI.
2. Inspect a live diagnostic response and record actual source image dates and retrieval time.
3. Confirm whether the default 45-day age limit permits timely monthly issuance; do not loosen it merely to make the endpoint return a number.
4. Add a durable append-only forecast archive before claiming prospective shadow evaluation.
5. Add UI handling for HTTP 503 so cached prior values are visibly marked stale and cannot appear as current.
6. Add source-vintage snapshots and revision-difference monitoring.
