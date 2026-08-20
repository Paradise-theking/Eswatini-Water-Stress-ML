from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

import ee
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATCHMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "mnjoli_w60e_catchment.geojson"
)

ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"


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
    "potential_evaporation_sum",
]


def initialize_earth_engine() -> None:
    """
    Initialize Earth Engine.

    Local development:
        Uses the authenticated local Earth Engine account.

    Production:
        Uses a service account when EE_SERVICE_ACCOUNT and
        GOOGLE_APPLICATION_CREDENTIALS are available.
    """

    import os

    service_account = os.getenv("EE_SERVICE_ACCOUNT")
    project_id = os.getenv(
        "EE_PROJECT_ID",
        "eswatiniwaterstress",
    )
    credentials_path = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )

    if service_account and credentials_path:
        credentials = ee.ServiceAccountCredentials(
            service_account,
            credentials_path,
        )

        ee.Initialize(
            credentials,
            project=project_id,
        )

        return

    ee.Initialize(
        project=project_id
    )

def get_region() -> ee.Geometry:
    """
    Load the exact Mnjoli W60E catchment geometry
    used by the research workflow.
    """

    if not CATCHMENT_PATH.exists():
        raise FileNotFoundError(
            f"Catchment GeoJSON not found: {CATCHMENT_PATH}"
        )

    with open(
        CATCHMENT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        geojson = json.load(f)

    geo_type = geojson.get("type")

    if geo_type == "FeatureCollection":
        features = geojson.get("features", [])

        if not features:
            raise ValueError(
                "Catchment FeatureCollection contains no features."
            )

        geometries = [
            feature["geometry"]
            for feature in features
            if feature.get("geometry")
        ]

        if not geometries:
            raise ValueError(
                "No valid geometries found in catchment GeoJSON."
            )

        if len(geometries) == 1:
            return ee.Geometry(geometries[0])

        ee_features = [
            ee.Feature(ee.Geometry(geometry))
            for geometry in geometries
        ]

        return (
            ee.FeatureCollection(ee_features)
            .geometry()
        )

    if geo_type == "Feature":
        return ee.Geometry(
            geojson["geometry"]
        )

    if geo_type in {
        "Polygon",
        "MultiPolygon",
    }:
        return ee.Geometry(geojson)

    raise ValueError(
        f"Unsupported GeoJSON type: {geo_type}"
    )


def latest_available_dates() -> Dict[str, str]:
    """Return latest available dates from ERA5-Land and CHIRPS."""

    era5_latest = (
        ee.ImageCollection(ERA5_COLLECTION)
        .sort("system:time_start", False)
        .first()
    )

    chirps_latest = (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .sort("system:time_start", False)
        .first()
    )

    return {
        "era5_land": era5_latest.date()
        .format("YYYY-MM-dd")
        .getInfo(),
        "chirps": chirps_latest.date()
        .format("YYYY-MM-dd")
        .getInfo(),
    }


def latest_common_complete_month() -> pd.Timestamp:
    """
    Determine the latest fully completed calendar month
    available in both ERA5-Land and CHIRPS.
    """

    dates = latest_available_dates()

    era5_date = pd.Timestamp(dates["era5_land"])
    chirps_date = pd.Timestamp(dates["chirps"])

    latest_common = min(
        era5_date,
        chirps_date,
    )

    month_end = latest_common + pd.offsets.MonthEnd(0)

    if latest_common.normalize() == month_end.normalize():
        return latest_common.to_period("M").to_timestamp()

    previous_month = (
        latest_common.to_period("M") - 1
    ).to_timestamp()

    return previous_month


def _reduce_image(
    image: ee.Image,
    bands: list[str],
) -> dict:
    """
    Spatially average ERA5 variables over the study region.
    """

    values = (
        image.select(bands)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=get_region(),
            scale=10000,
            bestEffort=True,
            maxPixels=1_000_000_000,
        )
        .getInfo()
    )

    values["date"] = (
        image.date()
        .format("YYYY-MM-dd")
        .getInfo()
    )

    return values


def fetch_era5_daily(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch ERA5-Land daily data.

    start_date is inclusive.
    end_date is exclusive.
    """

    collection = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterDate(
            start_date,
            end_date,
        )
        .select(ERA5_BANDS)
        .sort("system:time_start")
    )

    count = collection.size().getInfo()

    if count == 0:
        raise RuntimeError(
            "ERA5-Land request returned no images."
        )

    images = collection.toList(count)

    rows = []

    for i in range(count):
        image = ee.Image(
            images.get(i)
        )

        row = _reduce_image(
            image,
            ERA5_BANDS,
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "ERA5-Land request returned no data."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return (
        df.sort_values("date")
        .reset_index(drop=True)
    )


def fetch_chirps_daily(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch CHIRPS daily precipitation.

    start_date is inclusive.
    end_date is exclusive.
    """

    collection = (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .filterDate(
            start_date,
            end_date,
        )
        .select("precipitation")
        .sort("system:time_start")
    )

    count = collection.size().getInfo()

    if count == 0:
        raise RuntimeError(
            "CHIRPS request returned no images."
        )

    images = collection.toList(count)

    rows = []

    for i in range(count):
        image = ee.Image(
            images.get(i)
        )

        precipitation = (
            image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=get_region(),
                scale=5000,
                bestEffort=True,
                maxPixels=1_000_000_000,
            )
            .get("precipitation")
            .getInfo()
        )

        rows.append(
            {
                "date": (
                    image.date()
                    .format("YYYY-MM-dd")
                    .getInfo()
                ),
                "precipitation_mm": precipitation,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "CHIRPS request returned no data."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return (
        df.sort_values("date")
        .reset_index(drop=True)
    )


def transform_daily_data(
    era5: pd.DataFrame,
    chirps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the transformations used during model training.
    """

    df = pd.merge(
        chirps,
        era5,
        on="date",
        how="inner",
    )

    if df.empty:
        raise RuntimeError(
            "No matching dates between ERA5-Land and CHIRPS."
        )

    # -----------------------------------------
    # PET
    # -----------------------------------------

    df["pet_m"] = (
        -df["potential_evaporation_sum"]
    )

    df["pet_mm"] = (
        df["pet_m"] * 1000
    )

    # -----------------------------------------
    # Temperature
    # -----------------------------------------

    df["temperature_c"] = (
        df["temperature_2m"]
        - 273.15
    )

    df["temperature_min_c"] = (
        df["temperature_2m_min"]
        - 273.15
    )

    df["temperature_max_c"] = (
        df["temperature_2m_max"]
        - 273.15
    )

    df["dewpoint_c"] = (
        df["dewpoint_temperature_2m"]
        - 273.15
    )

    # -----------------------------------------
    # Wind speed
    # -----------------------------------------

    df["wind_speed"] = np.sqrt(
        df["u_component_of_wind_10m"] ** 2
        + df["v_component_of_wind_10m"] ** 2
    )

    # -----------------------------------------
    # Surface pressure
    # -----------------------------------------

    df["surface_pressure_kpa"] = (
        df["surface_pressure"]
        / 1000
    )

    return df


def aggregate_monthly(
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate daily observations to monthly values
    using the same rules used during training.
    """

    df = daily_df.copy()

    df["month_date"] = (
        df["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly = (
        df.groupby("month_date")
        .agg(
            precipitation_mm=(
                "precipitation_mm",
                "sum",
            ),
            pet_mm=(
                "pet_mm",
                "sum",
            ),
            temperature_mean_c=(
                "temperature_c",
                "mean",
            ),
            temperature_min_c=(
                "temperature_min_c",
                "mean",
            ),
            temperature_max_c=(
                "temperature_max_c",
                "mean",
            ),
            dewpoint_c=(
                "dewpoint_c",
                "mean",
            ),
            soil_moisture_layer1=(
                "volumetric_soil_water_layer_1",
                "mean",
            ),
            soil_moisture_layer2=(
                "volumetric_soil_water_layer_2",
                "mean",
            ),
            runoff_mm=(
                "runoff_sum",
                "sum",
            ),
            surface_runoff_mm=(
                "surface_runoff_sum",
                "sum",
            ),
            solar_radiation=(
                "surface_solar_radiation_downwards_sum",
                "mean",
            ),
            wind_speed=(
                "wind_speed",
                "mean",
            ),
            surface_pressure_kpa=(
                "surface_pressure_kpa",
                "mean",
            ),
        )
        .reset_index()
    )

    return monthly


def fetch_latest_monthly_data() -> pd.DataFrame:
    """
    Fetch enough recent data to calculate lagged
    and rolling features for the latest forecast.
    """

    latest_month = (
        latest_common_complete_month()
    )

    # We need at least:
    # current month
    # lag1
    # previous 3 months for rolling features
    #
    # Fetch 4 complete months for safety.
    start_month = (
        latest_month
        - pd.DateOffset(months=3)
    )

    end_date = (
        latest_month
        + pd.offsets.MonthEnd(1)
        + pd.Timedelta(days=1)
    )

    era5 = fetch_era5_daily(
        start_month.strftime(
            "%Y-%m-%d"
        ),
        end_date.strftime(
            "%Y-%m-%d"
        ),
    )

    chirps = fetch_chirps_daily(
        start_month.strftime(
            "%Y-%m-%d"
        ),
        end_date.strftime(
            "%Y-%m-%d"
        ),
    )

    daily = transform_daily_data(
        era5,
        chirps,
    )

    monthly = aggregate_monthly(
        daily
    )

    return monthly


if __name__ == "__main__":

    initialize_earth_engine()

    print(
        "\n========================================"
    )
    print(
        "CURRENT DATA INGESTION TEST"
    )
    print(
        "========================================"
    )

    dates = latest_available_dates()

    print("\nLatest data availability:")
    print(
        "ERA5-Land:",
        dates["era5_land"],
    )
    print(
        "CHIRPS:",
        dates["chirps"],
    )

    latest_month = (
        latest_common_complete_month()
    )

    print(
        "\nLatest common completed month:",
        latest_month.strftime(
            "%Y-%m-%d"
        ),
    )

    print(
        "\nFetching recent environmental data..."
    )

    monthly = (
        fetch_latest_monthly_data()
    )

    print(
        "\n========================================"
    )
    print(
        "MONTHLY DATA"
    )
    print(
        "========================================"
    )

    print(
        monthly.to_string(
            index=False
        )
    )

    print(
        "\nRows:",
        len(monthly),
    )

    print(
        "\nLatest row:"
    )

    print(
        monthly.tail(1).T
    )