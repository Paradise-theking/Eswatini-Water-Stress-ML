from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import subprocess
from typing import Any

import ee
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = "eswatiniwaterstress"
START_DATE = pd.Timestamp("2026-01-01")
HISTORICAL_MONTHLY = (
    ROOT
    / "data"
    / "processed"
    / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025.csv"
)
OUTPUT_ROOT = ROOT / "outputs" / "expanded" / "shadow_evaluation_data"
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_BANDS = (
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
)
SWBA_BASELINE_START = 1981
SWBA_BASELINE_END = 2010


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_earth_engine() -> None:
    try:
        ee.Initialize(project=PROJECT)
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize Google Earth Engine. Authenticate with "
            "`earthengine authenticate` and confirm project access."
        ) from exc


def get_common_complete_end() -> tuple[pd.Timestamp, dict[str, str]]:
    latest_dates: dict[str, str] = {}
    for name, collection_id in (
        ("CHIRPS", CHIRPS_COLLECTION),
        ("ERA5-Land", ERA5_COLLECTION),
    ):
        collection = ee.ImageCollection(collection_id).filterDate(
            START_DATE.strftime("%Y-%m-%d"),
            (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
        )
        latest_timestamp = collection.aggregate_max(
            "system:time_start"
        ).getInfo()
        if latest_timestamp is None:
            raise RuntimeError(f"No {name} imagery is available in 2026.")
        latest_date = ee.Date(latest_timestamp).format("YYYY-MM-dd").getInfo()
        latest_dates[name] = latest_date

    common_latest = min(pd.Timestamp(value) for value in latest_dates.values())
    month_end = common_latest + pd.offsets.MonthEnd(0)
    if month_end > common_latest:
        month_end -= pd.offsets.MonthEnd(1)
    if month_end < START_DATE:
        raise RuntimeError(
            "Neither dataset has a complete common month available in 2026."
        )
    return pd.Timestamp(month_end.date()), latest_dates


def make_catchment() -> Any:
    basins = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_12")
    point = ee.Geometry.Point([31.66667, -26.16667])
    candidates = basins.filterBounds(point)
    count = int(candidates.size().getInfo())
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one Mnjoli basin feature; Earth Engine returned "
            f"{count}."
        )
    return candidates.first().geometry()


def extract_collection(
    collection_id: str,
    catchment: Any,
    end_date: pd.Timestamp,
    source_name: str,
) -> pd.DataFrame:
    exclusive_end = end_date + pd.Timedelta(days=1)
    collection = (
        ee.ImageCollection(collection_id)
        .filterDate(
            START_DATE.strftime("%Y-%m-%d"),
            exclusive_end.strftime("%Y-%m-%d"),
        )
        .filterBounds(catchment)
        .sort("system:time_start")
    )
    if source_name == "CHIRPS":
        def extract_image(image: Any) -> Any:
            rainfall = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=catchment,
                scale=5566,
                bestEffort=True,
            ).get("precipitation")
            return ee.Feature(
                None,
                {
                    "date": image.date().format("YYYY-MM-dd"),
                    "precipitation_mm": rainfall,
                },
            )

        columns = ["date", "precipitation_mm"]
    else:
        def extract_image(image: Any) -> Any:
            values = image.select(list(ERA5_BANDS)).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=catchment,
                scale=11132,
                bestEffort=True,
            )
            properties = {"date": image.date().format("YYYY-MM-dd")}
            for band in ERA5_BANDS:
                properties[band] = values.get(band)
            return ee.Feature(None, properties)

        columns = ["date", *ERA5_BANDS]

    image_count = int(collection.size().getInfo())
    features = collection.map(extract_image).getInfo()
    rows = [feature["properties"] for feature in features["features"]]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        raise RuntimeError(f"{source_name} returned no daily observations.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)

    expected_dates = pd.date_range(START_DATE, end_date, freq="D")
    actual_dates = pd.DatetimeIndex(frame["date"])
    missing = expected_dates.difference(actual_dates)
    extra = actual_dates.difference(expected_dates)
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError(f"{source_name} contains invalid or duplicate dates.")
    if len(missing) or len(extra):
        raise ValueError(
            f"{source_name} daily coverage mismatch through "
            f"{end_date:%Y-%m-%d}: {len(missing)} missing, {len(extra)} extra."
        )
    if frame[columns[1:]].isna().any().any():
        raise ValueError(f"{source_name} contains missing daily measurements.")
    frame.attrs["image_count"] = image_count
    return frame


def aggregate_new_months(
    chirps: pd.DataFrame,
    era5: pd.DataFrame,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    daily = chirps.merge(
        era5,
        on="date",
        how="inner",
        validate="one_to_one",
    )
    if len(daily) != len(chirps) or len(daily) != len(era5):
        raise ValueError("CHIRPS and ERA5-Land daily dates do not align.")

    daily["pet_mm"] = -daily["potential_evaporation_sum"] * 1000
    daily["temperature_mean_c"] = daily["temperature_2m"] - 273.15
    daily["temperature_min_c"] = daily["temperature_2m_min"] - 273.15
    daily["temperature_max_c"] = daily["temperature_2m_max"] - 273.15
    daily["dewpoint_c"] = daily["dewpoint_temperature_2m"] - 273.15
    daily["water_balance_mm"] = (
        daily["precipitation_mm"] - daily["pet_mm"]
    )
    daily = daily.set_index("date")
    monthly = daily.resample("MS").agg(
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
    monthly = monthly.loc[monthly.index <= end_date]
    expected_months = pd.date_range(
        START_DATE, end_date, freq="MS"
    )
    if not monthly.index.equals(expected_months):
        raise ValueError("Refreshed monthly data has missing calendar months.")
    expected_days = monthly.index.days_in_month
    observed_days = daily.resample("MS").size().reindex(monthly.index)
    if not np.array_equal(observed_days.to_numpy(), expected_days.to_numpy()):
        raise ValueError("At least one refreshed calendar month is incomplete.")
    monthly.index.name = "month_date"
    return monthly.reset_index()


def build_updated_monthly(
    historical_path: Path,
    new_monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    historical = pd.read_csv(historical_path, parse_dates=["month_date"])
    if historical["month_date"].max() + pd.offsets.MonthBegin(1) != START_DATE:
        raise ValueError(
            "Historical monthly input must end immediately before "
            f"{START_DATE:%Y-%m}."
        )
    updated = pd.concat([historical, new_monthly], ignore_index=True)
    if updated["month_date"].duplicated().any():
        raise ValueError("Updated monthly data contains duplicate months.")
    updated = updated.sort_values("month_date").reset_index(drop=True)
    updated["year"] = updated["month_date"].dt.year
    updated["month"] = updated["month_date"].dt.month
    updated["water_balance_3month"] = (
        updated["water_balance_mm"].rolling(3, min_periods=3).sum()
    )
    updated["water_balance_6month"] = (
        updated["water_balance_mm"].rolling(6, min_periods=6).sum()
    )
    climatology = (
        updated.loc[
            updated["year"].between(
                SWBA_BASELINE_START, SWBA_BASELINE_END
            )
        ]
        .groupby("month")["water_balance_3month"]
        .agg(
            climatology_mean="mean",
            climatology_std="std",
            baseline_count="count",
        )
        .reset_index()
    )
    if climatology[["climatology_mean", "climatology_std"]].isna().any().any():
        raise ValueError("Could not calculate the frozen SWBA climatology.")
    updated = updated.merge(
        climatology,
        on="month",
        how="left",
        validate="many_to_one",
    )
    updated["swba_3month"] = (
        updated["water_balance_3month"] - updated["climatology_mean"]
    ) / updated["climatology_std"]
    conditions = [
        updated["swba_3month"] <= -2,
        updated["swba_3month"] <= -1,
        updated["swba_3month"] < 0,
        updated["swba_3month"] <= 1,
        updated["swba_3month"] <= 2,
        updated["swba_3month"] > 2,
    ]
    updated["swba_category"] = np.select(
        conditions,
        [
            "Exceptionally Dry",
            "Severely Dry",
            "Moderately Dry",
            "Near Normal",
            "Moderately Wet",
            "Exceptionally Wet",
        ],
        default="Unclassified",
    )
    return updated, historical


def main() -> None:
    initialize_earth_engine()
    end_date, latest_dates = get_common_complete_end()
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / f"refresh_through_{end_date:%Y-%m}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    catchment = make_catchment()
    chirps = extract_collection(
        CHIRPS_COLLECTION, catchment, end_date, "CHIRPS"
    )
    era5 = extract_collection(
        ERA5_COLLECTION, catchment, end_date, "ERA5-Land"
    )
    chirps_path = run_dir / "chirps_daily_refresh.csv"
    era5_path = run_dir / "era5_land_daily_refresh_raw.csv"
    chirps.to_csv(chirps_path, index=False, date_format="%Y-%m-%d")
    era5.to_csv(era5_path, index=False, date_format="%Y-%m-%d")

    new_monthly = aggregate_new_months(chirps, era5, end_date)
    updated, historical = build_updated_monthly(
        HISTORICAL_MONTHLY, new_monthly
    )
    raw_output = run_dir / "mnjoli_water_stress_monthly_updated.csv"
    swba_output = run_dir / "mnjoli_water_stress_monthly_updated_swba.csv"
    updated.drop(
        columns=[
            "climatology_mean",
            "climatology_std",
            "baseline_count",
            "swba_3month",
            "swba_category",
        ]
    ).to_csv(raw_output, index=False, date_format="%Y-%m-%d")
    updated.to_csv(swba_output, index=False, date_format="%Y-%m-%d")

    old_history = pd.read_csv(
        ROOT
        / "data"
        / "processed"
        / "expanded"
        / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
    )
    history_preserved = updated.iloc[: len(historical)].reset_index(drop=True)
    compared_columns = [
        column
        for column in old_history.columns
        if column in history_preserved.columns
        and column not in {"month_date", "swba_category"}
    ]
    if not np.allclose(
        old_history[compared_columns].to_numpy(dtype=float),
        history_preserved[compared_columns].to_numpy(dtype=float),
        rtol=0,
        atol=1e-8,
        equal_nan=True,
    ):
        raise ValueError(
            "Appending refreshed months changed the historical SWBA series."
        )

    audit_dir = run_dir / "data_audit"
    manifest = {
        "refresh_run_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "earth_engine_project": PROJECT,
        "sources": {
            "precipitation": CHIRPS_COLLECTION,
            "meteorological_and_land": ERA5_COLLECTION,
            "catchment": "WWF/HydroSHEDS/v1/Basins/hybas_12",
        },
        "source_latest_dates_at_refresh": latest_dates,
        "common_complete_data_through": end_date.strftime("%Y-%m-%d"),
        "extraction_start": START_DATE.strftime("%Y-%m-%d"),
        "extracted_days_per_source": len(chirps),
        "extracted_images": {
            "CHIRPS": chirps.attrs["image_count"],
            "ERA5-Land": era5.attrs["image_count"],
        },
        "historical_monthly_input": str(HISTORICAL_MONTHLY),
        "historical_monthly_input_sha256": sha256_file(HISTORICAL_MONTHLY),
        "historical_months_preserved": len(historical),
        "historical_values_preserved_within_1e-8": True,
        "updated_monthly_rows": len(updated),
        "updated_start_month": updated["month_date"].min().strftime("%Y-%m"),
        "updated_end_month": updated["month_date"].max().strftime("%Y-%m"),
        "outputs": {
            "chirps_daily": str(chirps_path),
            "era5_land_daily": str(era5_path),
            "monthly": str(raw_output),
            "monthly_swba": str(swba_output),
            "audit_dir": str(audit_dir),
        },
    }
    (run_dir / "refresh_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))

    audit_script = ROOT / "scripts" / "audit_drought_onset_data.py"
    audit_result = subprocess.run(
        [
            sys.executable,
            str(audit_script),
            "--input",
            str(swba_output),
            "--output-dir",
            str(audit_dir),
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if audit_result.returncode:
        raise RuntimeError(
            "Refreshed monthly SWBA data audit failed.\n"
            f"stdout:\n{audit_result.stdout}\n"
            f"stderr:\n{audit_result.stderr}"
        )
    audit_report = json.loads(
        (audit_dir / "audit_report.json").read_text(encoding="utf-8")
    )
    if not audit_report["all_checks_passed"]:
        raise RuntimeError("Refreshed monthly SWBA failed the onset data audit.")
    manifest["audit_passed"] = True
    manifest["audit_input_sha256"] = audit_report["input_sha256"]
    (run_dir / "refresh_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Refresh and audit passed through {end_date:%Y-%m}; "
        f"{audit_report['onsets']['onset_events']} onset events found; "
        f"outputs are under {run_dir}."
    )


if __name__ == "__main__":
    main()
