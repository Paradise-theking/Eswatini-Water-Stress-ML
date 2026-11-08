
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

START_YEAR = 1981
END_YEAR = 2025

DATA_EXTERNAL = Path("data/external/expanded")
DATA_PROCESSED = Path("data/processed/expanded")

CHIRPS_FILE = (
    DATA_EXTERNAL
    / f"mnjoli_chirps_daily_{START_YEAR}_{END_YEAR}.csv"
)

ERA5_FILE = (
    DATA_EXTERNAL
    / f"mnjoli_era5_land_daily_{START_YEAR}_{END_YEAR}_raw.csv"
)

OUTPUT_FILE = (
    DATA_PROCESSED
    / f"mnjoli_water_stress_monthly_{START_YEAR}_{END_YEAR}.csv"
)


# ============================================================
# LOAD DAILY DATA
# ============================================================

print("=" * 65)
print("LOADING EXPANDED DAILY DATA")
print("=" * 65)

chirps_df = pd.read_csv(CHIRPS_FILE, parse_dates=["date"])
era5_df = pd.read_csv(ERA5_FILE, parse_dates=["date"])

print("CHIRPS shape:", chirps_df.shape)
print("ERA5 shape:", era5_df.shape)


# ============================================================
# VALIDATE INPUTS
# ============================================================

if chirps_df["date"].duplicated().any():
    raise ValueError("Duplicate dates found in CHIRPS.")

if era5_df["date"].duplicated().any():
    raise ValueError("Duplicate dates found in ERA5.")

if not chirps_df["date"].equals(era5_df["date"]):
    raise ValueError("CHIRPS and ERA5 dates do not align.")

if chirps_df["precipitation_mm"].isna().any():
    raise ValueError("Missing CHIRPS precipitation values.")

if era5_df.isna().any().any():
    raise ValueError("Missing values found in ERA5 raw data.")


# ============================================================
# PREPARE CHIRPS
# ============================================================

chirps_clean = chirps_df[
    ["date", "precipitation_mm"]
].copy()


# ============================================================
# PREPARE ERA5-LAND
# ============================================================

era5_clean = era5_df[
    [
        "date",
        "temperature_2m",
        "temperature_2m_min",
        "temperature_2m_max",
        "dewpoint_temperature_2m",
        "surface_pressure",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "surface_solar_radiation_downwards_sum",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "runoff_sum",
        "surface_runoff_sum",
        "potential_evaporation_sum",
    ]
].copy()


# ============================================================
# MERGE DAILY DATASETS
# ============================================================

print("\nMerging CHIRPS and ERA5-Land...")

daily_df = pd.merge(
    chirps_clean,
    era5_clean,
    on="date",
    how="inner",
    validate="one_to_one",
)

daily_df = daily_df.sort_values("date").reset_index(drop=True)

print("Merged daily shape:", daily_df.shape)


# ============================================================
# DERIVE PET AND TEMPERATURE VARIABLES
# ============================================================

# ERA5 potential evaporation is negative by convention.
daily_df["pet_m"] = (
    -daily_df["potential_evaporation_sum"]
)

daily_df["pet_mm"] = (
    daily_df["pet_m"] * 1000
)

# Convert temperatures from Kelvin to Celsius.
daily_df["temperature_mean_c"] = (
    daily_df["temperature_2m"] - 273.15
)

daily_df["temperature_min_c"] = (
    daily_df["temperature_2m_min"] - 273.15
)

daily_df["temperature_max_c"] = (
    daily_df["temperature_2m_max"] - 273.15
)

daily_df["dewpoint_c"] = (
    daily_df["dewpoint_temperature_2m"] - 273.15
)


# ============================================================
# DAILY WATER BALANCE
# ============================================================

daily_df["water_balance_mm"] = (
    daily_df["precipitation_mm"]
    - daily_df["pet_mm"]
)


# ============================================================
# MONTHLY AGGREGATION
# ============================================================

print("\nAggregating daily values to monthly...")

daily_df = daily_df.set_index("date")

monthly_df = daily_df.resample("MS").agg(
    {
        "precipitation_mm": "sum",
        "pet_mm": "sum",
        "water_balance_mm": "sum",

        "temperature_mean_c": "mean",
        "temperature_min_c": "mean",
        "temperature_max_c": "mean",
        "dewpoint_c": "mean",

        "surface_pressure": "mean",
        "u_component_of_wind_10m": "mean",
        "v_component_of_wind_10m": "mean",

        "surface_solar_radiation_downwards_sum": "sum",

        "volumetric_soil_water_layer_1": "mean",
        "volumetric_soil_water_layer_2": "mean",

        "runoff_sum": "sum",
        "surface_runoff_sum": "sum",
    }
)


# ============================================================
# ROLLING WATER BALANCE
# ============================================================

monthly_df["water_balance_3month"] = (
    monthly_df["water_balance_mm"]
    .rolling(window=3, min_periods=3)
    .sum()
)

monthly_df["water_balance_6month"] = (
    monthly_df["water_balance_mm"]
    .rolling(window=6, min_periods=6)
    .sum()
)


# ============================================================
# ADD CALENDAR FEATURES
# ============================================================

monthly_df["year"] = monthly_df.index.year
monthly_df["month"] = monthly_df.index.month

monthly_df = monthly_df.reset_index()

monthly_df = monthly_df.rename(
    columns={"date": "month_date"}
)


# ============================================================
# VALIDATE MONTHLY DATA
# ============================================================

expected_months = (
    (END_YEAR - START_YEAR + 1) * 12
)

expected_dates = pd.date_range(
    start=f"{START_YEAR}-01-01",
    end=f"{END_YEAR}-12-01",
    freq="MS",
)

print("\n" + "=" * 65)
print("MONTHLY DATA VALIDATION")
print("=" * 65)

print("Rows:", len(monthly_df))
print("Expected months:", expected_months)

print(
    "Date range:",
    monthly_df["month_date"].min(),
    "to",
    monthly_df["month_date"].max(),
)

print(
    "Unique months:",
    monthly_df["month_date"].nunique(),
)

print(
    "Duplicate months:",
    monthly_df["month_date"].duplicated().sum(),
)

print(
    "Monthly dates complete:",
    monthly_df["month_date"].equals(
        pd.Series(expected_dates, name="month_date")
    ),
)

print("\nMissing values by column:")
print(monthly_df.isna().sum().to_string())


# ============================================================
# SAVE EXPANDED MONTHLY DATASET
# ============================================================

DATA_PROCESSED.mkdir(
    parents=True,
    exist_ok=True,
)

monthly_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 65)
print("MONTHLY AGGREGATION COMPLETE")
print("=" * 65)

print("Saved to:", OUTPUT_FILE)
print("Final shape:", monthly_df.shape)
