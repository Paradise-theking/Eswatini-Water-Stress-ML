from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
from statistics import fmean
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESERVOIR_BOUNDARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "mnjoli_reservoir_boundary.geojson"
)
CATCHMENT_AREA_KM2 = 896.9
FORECAST_DAYS = 30
SENTINEL_LOOKBACK_DAYS = 90
MIN_VALID_PIXEL_COVERAGE = 0.8


class WaterRiskDataError(RuntimeError):
    """Raised when an external water-risk source cannot provide valid data."""


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile from no values.")

    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]

    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_weather_ensemble(
    provider_data: dict,
    retrieved_at: datetime | None = None,
) -> dict:
    """Summarize paired precipitation and ET0 totals across forecast members."""
    daily = provider_data.get("daily")
    dates = daily.get("time") if isinstance(daily, dict) else None
    if not isinstance(dates, list) or len(dates) != FORECAST_DAYS:
        raise WaterRiskDataError(
            f"Weather provider must return {FORECAST_DAYS} daily values."
        )

    try:
        for date in dates:
            datetime.strptime(date, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise WaterRiskDataError(
            "Weather provider returned an invalid forecast date."
        ) from exc

    precipitation_fields = {
        int(match.group(1)): field
        for field in daily
        if (match := re.fullmatch(r"precipitation_sum_member(\d+)", field))
    }
    et0_fields = {
        int(match.group(1)): field
        for field in daily
        if (match := re.fullmatch(
            r"et0_fao_evapotranspiration_member(\d+)",
            field,
        ))
    }
    member_ids = sorted(precipitation_fields.keys() & et0_fields.keys())
    member_totals = []

    for member_id in member_ids:
        precipitation = daily[precipitation_fields[member_id]]
        et0 = daily[et0_fields[member_id]]
        if (
            not isinstance(precipitation, list)
            or not isinstance(et0, list)
            or len(precipitation) != FORECAST_DAYS
            or len(et0) != FORECAST_DAYS
        ):
            continue

        values = precipitation + et0
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            continue

        precipitation_total = float(sum(precipitation))
        et0_total = float(sum(et0))
        member_totals.append({
            "precipitation_mm": precipitation_total,
            "reference_et0_mm": et0_total,
            "net_climate_balance_mm": precipitation_total - et0_total,
        })

    if len(member_totals) < 10:
        raise WaterRiskDataError(
            "Weather provider returned fewer than 10 complete paired ensemble members."
        )

    def summarize(field: str) -> dict[str, float]:
        values = [member[field] for member in member_totals]
        return {
            "p10": _percentile(values, 10),
            "median": _percentile(values, 50),
            "p90": _percentile(values, 90),
            "ensemble_mean": fmean(values),
        }

    precipitation = summarize("precipitation_mm")
    et0 = summarize("reference_et0_mm")
    climate_balance = summarize("net_climate_balance_mm")

    return {
        "status": "success",
        "source": "Open-Meteo seasonal forecast ensemble",
        "model_context": (
            "The 30-day lead falls within ECMWF EC46 extended-range guidance; "
            "provider ensemble members are summarized, not locally bias-corrected."
        ),
        "retrieved_at": (retrieved_at or datetime.now(timezone.utc)).isoformat(),
        "timezone": provider_data.get("timezone", "Africa/Mbabane"),
        "requested_location": {
            "latitude": -26.15,
            "longitude": 31.66,
            "catchment_area_km2": CATCHMENT_AREA_KM2,
        },
        "resolved_grid_point": {
            "latitude": provider_data.get("latitude"),
            "longitude": provider_data.get("longitude"),
            "elevation_m": provider_data.get("elevation"),
        },
        "forecast_period": {
            "start_date": dates[0],
            "end_date": dates[-1],
            "days": len(dates),
        },
        "ensemble_member_count": len(member_totals),
        "precipitation_total_mm": precipitation,
        "reference_et0_total_mm": et0,
        "net_climate_balance_mm": climate_balance,
        "gross_rainfall_volume_ml": {
            key: value * CATCHMENT_AREA_KM2
            for key, value in precipitation.items()
        },
        "interpretation": (
            "Precipitation minus reference ET0 is a catchment-scale "
            "hydroclimate indicator, not reservoir inflow, runoff, storage, "
            "or a water-supply forecast. Gross rainfall volume is rainfall "
            "over the catchment before infiltration, runoff, routing, or "
            "reservoir operations."
        ),
    }


def fetch_weather_balance() -> dict:
    params = urlencode({
        "latitude": -26.15,
        "longitude": 31.66,
        "daily": "precipitation_sum,et0_fao_evapotranspiration",
        "forecast_days": FORECAST_DAYS,
        "timezone": "Africa/Mbabane",
    })
    url = f"https://seasonal-api.open-meteo.com/v1/seasonal?{params}"

    try:
        with urlopen(url, timeout=30) as response:
            provider_data = json.loads(response.read())
    except (URLError, TimeoutError) as exc:
        raise WaterRiskDataError(
            f"Open-Meteo seasonal forecast request failed: {exc}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WaterRiskDataError(
            "Open-Meteo returned an unreadable seasonal forecast response."
        ) from exc

    if not isinstance(provider_data, dict):
        raise WaterRiskDataError(
            "Open-Meteo returned an invalid seasonal forecast response."
        )

    return summarize_weather_ensemble(provider_data)


def summarize_surface_water_observations(
    observations: list[dict],
    retrieved_at: datetime | None = None,
) -> dict:
    """Select the two most recent sufficiently observed satellite scenes."""
    by_date: dict[str, dict] = {}
    for observation in observations:
        date = observation.get("date")
        try:
            datetime.strptime(date, "%Y-%m-%d")
            water_area_m2 = float(observation["water_area_m2"])
            valid_coverage_fraction = float(
                observation["valid_coverage_fraction"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WaterRiskDataError(
                "Satellite processing returned an invalid observation."
            ) from exc

        if (
            not math.isfinite(water_area_m2)
            or not math.isfinite(valid_coverage_fraction)
            or water_area_m2 < 0
            or not 0 <= valid_coverage_fraction <= 1
        ):
            raise WaterRiskDataError(
                "Satellite processing returned out-of-range observation values."
            )
        if valid_coverage_fraction < MIN_VALID_PIXEL_COVERAGE:
            continue

        candidate = {
            "date": date,
            "water_area_km2": water_area_m2 / 1_000_000,
            "valid_coverage_percent": valid_coverage_fraction * 100,
        }
        prior = by_date.get(date)
        if (
            prior is None
            or candidate["valid_coverage_percent"]
            > prior["valid_coverage_percent"]
        ):
            by_date[date] = candidate

    selected = sorted(
        by_date.values(),
        key=lambda item: item["date"],
        reverse=True,
    )[:2]
    if not selected:
        raise WaterRiskDataError(
            "No recent Sentinel-2 scene had at least 80% valid coverage "
            "of the mapped reservoir boundary."
        )

    latest = selected[0]
    previous = selected[1] if len(selected) > 1 else None
    area_change_percent = None
    if previous and previous["water_area_km2"] > 0:
        area_change_percent = (
            (latest["water_area_km2"] - previous["water_area_km2"])
            / previous["water_area_km2"]
            * 100
        )

    return {
        "status": "success",
        "source": "Copernicus Sentinel-2 surface reflectance via Google Earth Engine",
        "retrieved_at": (retrieved_at or datetime.now(timezone.utc)).isoformat(),
        "method": (
            "MNDWI > 0 within the OpenStreetMap Mnjoli Dam water polygon; "
            "cloud, shadow, snow, and unclassified pixels excluded using SCL."
        ),
        "minimum_valid_coverage_percent": MIN_VALID_PIXEL_COVERAGE * 100,
        "latest_observation": latest,
        "previous_observation": previous,
        "area_change_percent_vs_previous_scene": area_change_percent,
        "validation_status": "not validated against dated reservoir storage observations",
        "interpretation": (
            "Satellite water surface area is a spatial extent proxy only. "
            "It is not water level or storage volume, and the OSM polygon is "
            "not a surveyed maximum-pool boundary."
        ),
    }


def fetch_surface_water_area() -> dict:
    import ee

    from backend.data_ingestion import initialize_earth_engine

    with RESERVOIR_BOUNDARY_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as boundary_file:
        boundary = json.load(boundary_file)
    geometry = ee.Geometry(boundary["geometry"])

    initialize_earth_engine()
    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=SENTINEL_LOOKBACK_DAYS)).strftime(
        "%Y-%m-%d"
    )
    end_date = now.strftime("%Y-%m-%d")
    geometry_area_m2 = geometry.area(maxError=1)

    def summarize_image(image):
        scene_classification = image.select("SCL")
        valid_pixels = (
            scene_classification.eq(4)
            .Or(scene_classification.eq(5))
            .Or(scene_classification.eq(6))
        )
        mndwi = image.normalizedDifference(["B3", "B11"])
        water_pixels = valid_pixels.And(mndwi.gt(0))

        valid_area = (
            ee.Image.pixelArea()
            .rename("area")
            .updateMask(valid_pixels)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=20,
                maxPixels=1_000_000_000,
            )
            .get("area")
        )
        water_area = (
            ee.Image.pixelArea()
            .rename("area")
            .updateMask(water_pixels)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=20,
                maxPixels=1_000_000_000,
            )
            .get("area")
        )
        valid_area = ee.Number(ee.Algorithms.If(valid_area, valid_area, 0))
        water_area = ee.Number(ee.Algorithms.If(water_area, water_area, 0))

        return ee.Feature(None, {
            "date": image.date().format("YYYY-MM-dd"),
            "water_area_m2": water_area,
            "valid_coverage_fraction": valid_area.divide(geometry_area_m2),
        })

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .sort("system:time_start", False)
        .limit(100)
    )
    scene_list = collection.toList(collection.size())
    features = ee.FeatureCollection(
        scene_list.map(
            lambda image: summarize_image(ee.Image(image))
        )
    )
    result = features.getInfo()
    observations = [
        feature.get("properties", {})
        for feature in result.get("features", [])
    ]

    summary = summarize_surface_water_observations(observations, now)
    summary["boundary"] = {
        "name": boundary["properties"].get("name", "Mnjoli Dam"),
        "source": boundary["properties"].get("source", "OpenStreetMap"),
        "osm_relation_id": boundary["properties"].get("osm_id"),
        "attribution": boundary["properties"].get(
            "attribution",
            "© OpenStreetMap contributors",
        ),
    }
    summary["lookback_days"] = SENTINEL_LOOKBACK_DAYS
    return summary
