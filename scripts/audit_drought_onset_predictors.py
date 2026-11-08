
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "expanded"
    / "historical_drought_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "month_date"
SWBA_COL = "swba_3month"
DROUGHT_THRESHOLD = -1.0

# Variables available at the forecast origin.
PREDICTOR_VARIABLES = [
    "precipitation_mm",
    "pet_mm",
    "water_balance_mm",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "water_balance_3month",
    "water_balance_6month",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_min_c",
    "dewpoint_c",
    "surface_solar_radiation_downwards_sum",
    "runoff_sum",
    "surface_runoff_sum",
    SWBA_COL,
]

CHANGE_LAGS = [1, 3, 6]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [DATE_COL, SWBA_COL]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    errors="coerce"
)

df[SWBA_COL] = pd.to_numeric(
    df[SWBA_COL],
    errors="coerce"
)

df = (
    df.dropna(subset=[DATE_COL])
    .sort_values(DATE_COL)
    .reset_index(drop=True)
)

if df[DATE_COL].duplicated().any():
    raise ValueError("Duplicate monthly dates detected.")

# Keep only predictor variables present in the source.
available_predictors = [
    column for column in PREDICTOR_VARIABLES
    if column in df.columns
]

missing_predictors = [
    column for column in PREDICTOR_VARIABLES
    if column not in df.columns
]

for column in available_predictors:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

data = df.set_index(DATE_COL).sort_index()

print("=" * 70)
print("DROUGHT-ONSET PREDICTOR AUDIT")
print("=" * 70)

print(f"Monthly records: {len(df)}")
print(f"Available predictors: {len(available_predictors)}")

if missing_predictors:
    print(
        "Predictors not found in source:",
        missing_predictors
    )


# ============================================================
# 3. IDENTIFY DROUGHT ONSETS
# ============================================================

# A drought onset is the first month with SWBA <= -1
# immediately following a valid month with SWBA > -1.

onset_dates = []

for target_date, row in data.iterrows():

    target_swba = row[SWBA_COL]

    if pd.isna(target_swba):
        continue

    if target_swba > DROUGHT_THRESHOLD:
        continue

    origin_date = (
        target_date - pd.DateOffset(months=1)
    )

    if origin_date not in data.index:
        continue

    origin_swba = data.loc[
        origin_date, SWBA_COL
    ]

    # Do not classify an onset if the preceding
    # month's SWBA is missing.
    if pd.isna(origin_swba):
        continue

    if origin_swba > DROUGHT_THRESHOLD:
        onset_dates.append(target_date)


print(
    f"\nDrought onsets identified: {len(onset_dates)}"
)


# ============================================================
# 4. EXTRACT PRE-ONSET PREDICTORS
# ============================================================

records = []

for episode_number, target_date in enumerate(
    onset_dates,
    start=1
):

    origin_date = (
        target_date - pd.DateOffset(months=1)
    )

    origin = data.loc[origin_date]
    target = data.loc[target_date]

    record = {
        "episode_id": episode_number,
        "forecast_origin_month": origin_date,
        "drought_onset_month": target_date,
        "origin_swba": origin[SWBA_COL],
        "onset_observed_swba": target[SWBA_COL],
    }

    # Extract climate and soil conditions available
    # at the forecast origin.
    for variable in available_predictors:

        record[
            f"origin_{variable}"
        ] = origin[variable]

    # Calculate changes relative to earlier months.
    # Change = value at origin minus value at lagged month.
    for variable in available_predictors:

        for lag in CHANGE_LAGS:

            earlier_date = (
                origin_date
                - pd.DateOffset(months=lag)
            )

            if earlier_date in data.index:

                earlier_value = data.loc[
                    earlier_date, variable
                ]

                current_value = origin[variable]

                if (
                    pd.notna(earlier_value)
                    and pd.notna(current_value)
                ):
                    change = (
                        current_value - earlier_value
                    )
                else:
                    change = np.nan

            else:
                change = np.nan

            record[
                f"{variable}_change_{lag}m"
            ] = change

    # Describe the drought episode after onset.
    # These are outcome descriptors, NOT predictors.
    episode_values = []

    current_date = target_date

    while current_date in data.index:

        current_swba = data.loc[
            current_date, SWBA_COL
        ]

        if (
            pd.isna(current_swba)
            or current_swba > DROUGHT_THRESHOLD
        ):
            break

        episode_values.append(current_swba)

        current_date = (
            current_date + pd.DateOffset(months=1)
        )

    record["episode_end_month"] = (
        current_date - pd.DateOffset(months=1)
    )

    record["episode_duration_months"] = len(
        episode_values
    )

    record["episode_minimum_swba"] = min(
        episode_values
    )

    record["episode_mean_swba"] = np.mean(
        episode_values
    )

    records.append(record)


# ============================================================
# 5. CREATE AND SAVE ONSET TABLE
# ============================================================

audit = pd.DataFrame(records)

if audit.empty:
    raise ValueError(
        "No drought onsets were identified. "
        "Check the SWBA data and threshold."
    )

audit = audit.sort_values(
    "drought_onset_month"
).reset_index(drop=True)

output_file = (
    OUTPUT_DIR
    / "drought_onset_predictor_audit.csv"
)

audit.to_csv(
    output_file,
    index=False
)


# ============================================================
# 6. CREATE A COMPACT SUMMARY TABLE
# ============================================================

summary_variables = [
    f"origin_{variable}"
    for variable in available_predictors
]

summary_variables += [
    f"{variable}_change_{lag}m"
    for variable in available_predictors
    for lag in CHANGE_LAGS
]

summary_records = []

for column in summary_variables:

    values = pd.to_numeric(
        audit[column],
        errors="coerce"
    )

    if values.notna().sum() == 0:
        continue

    summary_records.append({
        "variable": column,
        "valid_onsets": values.notna().sum(),
        "mean": values.mean(),
        "median": values.median(),
        "minimum": values.min(),
        "maximum": values.max(),
    })

summary = pd.DataFrame(summary_records)

summary_file = (
    OUTPUT_DIR
    / "drought_onset_predictor_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 7. PRINT EVENT-BY-EVENT RESULTS
# ============================================================

display_columns = [
    "episode_id",
    "forecast_origin_month",
    "drought_onset_month",
    "origin_swba",
    "onset_observed_swba",
    "episode_duration_months",
    "episode_minimum_swba",
]

print("\n" + "=" * 70)
print("DROUGHT ONSET SUMMARY")
print("=" * 70)

print(
    audit[display_columns].to_string(
        index=False
    )
)


# ============================================================
# 8. PRINT SELECTED PRE-ONSET PREDICTORS
# ============================================================

selected_columns = [
    "forecast_origin_month",
    "drought_onset_month",
    "origin_precipitation_mm",
    "origin_pet_mm",
    "origin_water_balance_mm",
    "origin_volumetric_soil_water_layer_1",
    "origin_volumetric_soil_water_layer_2",
    "origin_water_balance_3month",
    "origin_water_balance_6month",
    "origin_swba",
]

selected_columns = [
    column for column in selected_columns
    if column in audit.columns
]

print("\n" + "=" * 70)
print("SELECTED CONDITIONS BEFORE DROUGHT ONSET")
print("=" * 70)

print(
    audit[selected_columns].round(4).to_string(
        index=False
    )
)


# ============================================================
# 9. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("DROUGHT-ONSET PREDICTOR AUDIT COMPLETED")
print("=" * 70)

print(f"Number of onset events: {len(audit)}")

print(
    "Mean drought duration:",
    round(
        audit["episode_duration_months"].mean(),
        2
    ),
    "months"
)

print(
    "Most severe onset episode SWBA:",
    round(
        audit["episode_minimum_swba"].min(),
        4
    )
)

print("\nSaved files:")
print(output_file)
print(summary_file)

print("\nIMPORTANT:")
print(
    "Predictors are extracted from the month BEFORE "
    "drought onset."
)

print(
    "Onset-month observations are used only to describe "
    "the observed outcome."
)
