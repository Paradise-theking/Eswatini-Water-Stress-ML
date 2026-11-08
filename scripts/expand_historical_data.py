from pathlib import Path
import sys
import re
import os
import pandas as pd
import ee

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = "eswatiniwaterstress"

START_YEAR = int(os.environ.get("EXPANSION_START_YEAR", "1981"))
END_YEAR = int(os.environ.get("EXPANSION_END_YEAR", "2025"))

OUTPUT_EXTERNAL = Path("data/external/expanded")
OUTPUT_PROCESSED = Path("data/processed/expanded")

OUTPUT_EXTERNAL.mkdir(parents=True, exist_ok=True)
OUTPUT_PROCESSED.mkdir(parents=True, exist_ok=True)

CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"

ERA5_BANDS = [
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
    "potential_evaporation_sum"
]


# ============================================================
# EARTH ENGINE INITIALIZATION
# ============================================================

print("\nInitializing Google Earth Engine...")

try:
    ee.Initialize(project=PROJECT)
except Exception as exc:
    print("\nEarth Engine initialization failed.")
    print("If necessary, authenticate with:")
    print("earthengine authenticate")
    print("\nError:", exc)
    sys.exit(1)

print("Earth Engine initialized.")


# ============================================================
# RECREATE ORIGINAL MNJOLI CATCHMENT
# ============================================================

print("\nLoading original Mnjoli catchment...")

hydrobasins = ee.FeatureCollection(
    "WWF/HydroSHEDS/v1/Basins/hybas_12"
)

mnjoli_point = ee.Geometry.Point([
    31.66667,
    -26.16667
])

candidate_basins = hydrobasins.filterBounds(mnjoli_point)

catchment_feature = candidate_basins.first()
catchment = catchment_feature.geometry()

candidate_count = candidate_basins.size().getInfo()
area_km2 = catchment.area().divide(1e6).getInfo()

print("Candidate basins:", candidate_count)
print("Catchment area (km²):", area_km2)

if candidate_count == 0:
    raise RuntimeError("No Mnjoli catchment was found.")

if candidate_count != 1:
    print(
        "WARNING: More than one candidate basin was found. "
        "The original notebook uses candidate_basins.first()."
    )


# ============================================================
# CHIRPS EXTRACTION
# ============================================================

def extract_chirps_year(year):

    output_file = (
        OUTPUT_EXTERNAL /
        f"mnjoli_chirps_daily_{year}.csv"
    )

    if output_file.exists():
        print(f"CHIRPS {year}: existing file skipped.")
        return

    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"

    collection = (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .filterDate(start_date, end_date)
        .filterBounds(catchment)
    )

    image_count = collection.size().getInfo()

    print(
        f"CHIRPS {year}: "
        f"{image_count} images available."
    )

    def extract_image(image):

        rainfall = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=catchment,
            scale=5566,
            bestEffort=True
        ).get("precipitation")

        return ee.Feature(
            None,
            {
                "date": image.date().format("YYYY-MM-dd"),
                "precipitation_mm": rainfall
            }
        )

    features = collection.map(extract_image).getInfo()

    rows = [
        feature["properties"]
        for feature in features["features"]
    ]

    df = pd.DataFrame(
        rows,
        columns=["date", "precipitation_mm"]
    )

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["precipitation_mm"] = pd.to_numeric(
            df["precipitation_mm"],
            errors="coerce"
        )
        df = df.sort_values("date").reset_index(drop=True)

    df.to_csv(output_file, index=False)

    print(
        f"CHIRPS {year}: saved "
        f"{len(df)} records."
    )


# ============================================================
# ERA5-LAND EXTRACTION
# ============================================================

def extract_era5_year(year):

    output_file = (
        OUTPUT_EXTERNAL /
        f"mnjoli_era5_land_daily_{year}_raw.csv"
    )

    if output_file.exists():
        print(f"ERA5-Land {year}: existing file skipped.")
        return

    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"

    collection = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterDate(start_date, end_date)
        .filterBounds(catchment)
    )

    image_count = collection.size().getInfo()

    print(
        f"ERA5-Land {year}: "
        f"{image_count} images available."
    )

    def extract_image(image):

        values = image.select(ERA5_BANDS).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=catchment,
            scale=11132,
            bestEffort=True
        )

        properties = {
            "date": image.date().format("YYYY-MM-dd")
        }

        for band in ERA5_BANDS:
            properties[band] = values.get(band)

        return ee.Feature(None, properties)

    features = collection.map(extract_image).getInfo()

    rows = [
        feature["properties"]
        for feature in features["features"]
    ]

    columns = ["date"] + ERA5_BANDS

    df = pd.DataFrame(rows, columns=columns)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

        numeric_columns = [
            col for col in df.columns
            if col != "date"
        ]

        df[numeric_columns] = df[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        df = df.sort_values("date").reset_index(drop=True)

    df.to_csv(output_file, index=False)

    print(
        f"ERA5-Land {year}: saved "
        f"{len(df)} records."
    )


# ============================================================
# YEAR-BY-YEAR EXTRACTION
# ============================================================

for year in range(START_YEAR, END_YEAR + 1):

    print("\n" + "=" * 60)
    print(f"PROCESSING YEAR {year}")
    print("=" * 60)

    extract_chirps_year(year)
    extract_era5_year(year)


# ============================================================
# COMBINE ANNUAL FILES
# ============================================================

def combine_annual_files(pattern, output_file):

    # Select annual files only; exclude previously combined datasets.
    candidates = OUTPUT_EXTERNAL.glob(pattern)

    files = sorted(
        file for file in candidates
        if re.search(r"_(19|20)\d{2}(?:_raw)?$", file.stem)
        and not re.search(r"\d{4}_\d{4}", file.stem)
    )

    expected_years = END_YEAR - START_YEAR + 1

    if len(files) != expected_years:
        print(
            f"\nCannot combine {pattern}: "
            f"found {len(files)} of "
            f"{expected_years} annual files."
        )
        return None

    frames = [
        pd.read_csv(file, parse_dates=["date"])
        for file in files
    ]

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    combined = combined.sort_values(
        "date"
    ).reset_index(drop=True)

    combined.to_csv(output_file, index=False)

    print("\nCombined dataset:", output_file)
    print("Shape:", combined.shape)
    print("Start:", combined["date"].min())
    print("End:", combined["date"].max())
    print("Duplicate dates:", combined["date"].duplicated().sum())

    return combined


print("\n" + "=" * 60)
print("COMBINING EXPANDED DATASETS")
print("=" * 60)

chirps_df = combine_annual_files(
    "mnjoli_chirps_daily_*.csv",
    OUTPUT_EXTERNAL /
    f"mnjoli_chirps_daily_{START_YEAR}_{END_YEAR}.csv"
)

era5_raw_df = combine_annual_files(
    "mnjoli_era5_land_daily_*_raw.csv",
    OUTPUT_EXTERNAL /
    f"mnjoli_era5_land_daily_{START_YEAR}_{END_YEAR}_raw.csv"
)


# ============================================================
# PET CONVERSION
# ============================================================

if era5_raw_df is not None:

    era5_processed_df = era5_raw_df.copy()

    era5_processed_df["pet_m"] = (
        -era5_processed_df["potential_evaporation_sum"]
    )

    era5_processed_df["pet_mm"] = (
        era5_processed_df["pet_m"] * 1000
    )

    processed_file = (
        OUTPUT_PROCESSED /
        f"mnjoli_era5_land_daily_{START_YEAR}_{END_YEAR}_processed.csv"
    )

    era5_processed_df.to_csv(
        processed_file,
        index=False
    )

    print("\nProcessed ERA5-Land saved:", processed_file)


# ============================================================
# DATE CONTINUITY AUDIT
# ============================================================

def audit_dates(df, name):

    if df is None:
        return

    print("\n" + "-" * 55)
    print(name)
    print("-" * 55)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    expected_dates = pd.date_range(
        f"{START_YEAR}-01-01",
        f"{END_YEAR}-12-31",
        freq="D"
    )

    actual_dates = pd.DatetimeIndex(
        df["date"].dropna().unique()
    ).sort_values()

    missing_dates = expected_dates.difference(actual_dates)

    print("Rows:", len(df))
    print("Unique dates:", len(actual_dates))
    print("Missing dates:", len(missing_dates))
    print("Duplicate dates:", df["date"].duplicated().sum())

    if len(missing_dates):
        print("First missing dates:")
        print(list(missing_dates[:10]))

    print(
        "Expected daily records:",
        len(expected_dates)
    )


audit_dates(chirps_df, "CHIRPS DATE AUDIT")
audit_dates(era5_raw_df, "ERA5-LAND DATE AUDIT")

print("\nEXPANSION EXTRACTION PROCESS FINISHED.")
