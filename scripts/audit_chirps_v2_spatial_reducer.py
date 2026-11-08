from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform


ROOT = Path(__file__).resolve().parents[1]
PROJECT = "eswatiniwaterstress"
START_YEAR = 1981
END_YEAR = 2025
CELL_SIZE_DEGREES = 0.05
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
V2_DAILY = (
    ROOT
    / "data"
    / "external"
    / "expanded"
    / "mnjoli_chirps_daily_1981_2025.csv"
)
CATCHMENT_GEOJSON = ROOT / "data" / "external" / "mnjoli_w60e_catchment.geojson"
OUTPUT_ROOT = (
    ROOT / "outputs" / "expanded" / "chirps_v3_transition"
    / "run_through_2025_20261106T181031Z"
)
OUTPUT_DIR = OUTPUT_ROOT / "v2_spatial_reducer_audit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_intersecting_cells(
    catchment: Any,
) -> tuple[pd.DataFrame, float, str]:
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32736", always_xy=True
    )
    project = transformer.transform
    projected_catchment = transform(project, catchment)
    half_cell = CELL_SIZE_DEGREES / 2

    longitude_centres = (
        -180
        + half_cell
        + np.arange(round(360 / CELL_SIZE_DEGREES))
        * CELL_SIZE_DEGREES
    )
    latitude_centres = (
        50
        - half_cell
        - np.arange(round(100 / CELL_SIZE_DEGREES))
        * CELL_SIZE_DEGREES
    )
    longitude_centres = longitude_centres[
        (longitude_centres + half_cell > catchment.bounds[0])
        & (longitude_centres - half_cell < catchment.bounds[2])
    ]
    latitude_centres = latitude_centres[
        (latitude_centres + half_cell > catchment.bounds[1])
        & (latitude_centres - half_cell < catchment.bounds[3])
    ]

    cells: list[dict[str, float | int]] = []
    cell_id = 0
    for latitude in latitude_centres:
        for longitude in longitude_centres:
            projected_cell = transform(
                project,
                box(
                    longitude - half_cell,
                    latitude - half_cell,
                    longitude + half_cell,
                    latitude + half_cell,
                ),
            )
            area = float(projected_catchment.intersection(projected_cell).area)
            if area > 0:
                cells.append(
                    {
                        "cell_id": cell_id,
                        "longitude": float(longitude),
                        "latitude": float(latitude),
                        "catchment_overlap_area_m2": area,
                    }
                )
                cell_id += 1

    weights = pd.DataFrame(cells)
    if weights.empty:
        raise ValueError("No CHIRPS v2 grid cells intersect the catchment.")
    catchment_area = float(projected_catchment.area)
    area_coverage = float(weights["catchment_overlap_area_m2"].sum()) / (
        catchment_area
    )
    if not np.isclose(area_coverage, 1.0, rtol=0, atol=1e-6):
        raise ValueError(
            "The selected CHIRPS v2 grid cells do not cover the catchment "
            f"(area ratio {area_coverage:.9f})."
        )
    weights["catchment_area_fraction"] = (
        weights["catchment_overlap_area_m2"] / catchment_area
    )
    return weights, catchment_area, f"{area_coverage:.12f}"


def initialize_earth_engine() -> None:
    try:
        ee.Initialize(project=PROJECT)
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize Google Earth Engine. Authenticate with "
            "`earthengine authenticate` and confirm project access."
        ) from exc


def load_v2_native_grid_samples(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    points = ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point([row.longitude, row.latitude]),
                {"cell_id": int(row.cell_id)},
            )
            for row in weights.itertuples(index=False)
        ]
    )
    frames: list[pd.DataFrame] = []
    cells_per_day = len(weights)

    for year in range(START_YEAR, END_YEAR + 1):
        start = f"{year}-01-01"
        exclusive_end = f"{year + 1}-01-01"
        collection = (
            ee.ImageCollection(CHIRPS_COLLECTION)
            .filterDate(start, exclusive_end)
            .sort("system:time_start")
        )
        image_count = int(collection.size().getInfo())
        expected_dates = pd.date_range(start, exclusive_end, inclusive="left")
        if image_count != len(expected_dates):
            raise ValueError(
                f"CHIRPS v2 returned {image_count} images in {year}; "
                f"expected {len(expected_dates)}."
            )
        first_image = ee.Image(collection.first()).select("precipitation")
        projection = first_image.projection()

        def sample_image(image: Any) -> Any:
            image = ee.Image(image).select("precipitation")
            return image.sampleRegions(
                collection=points,
                projection=projection,
                geometries=False,
            ).map(
                lambda feature: feature.set(
                    "date", image.date().format("YYYY-MM-dd")
                )
            )

        sampled = ee.FeatureCollection(collection.map(sample_image)).flatten()
        features = sampled.getInfo()["features"]
        rows = [feature["properties"] for feature in features]
        frame = pd.DataFrame(rows)
        required = {"cell_id", "date", "precipitation"}
        if not required.issubset(frame.columns):
            raise ValueError(
                f"CHIRPS v2 native sampling for {year} returned columns "
                f"{frame.columns.tolist()}."
            )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["cell_id"] = pd.to_numeric(frame["cell_id"], errors="coerce")
        frame["precipitation"] = pd.to_numeric(
            frame["precipitation"], errors="coerce"
        )
        if frame[["date", "cell_id", "precipitation"]].isna().any().any():
            raise ValueError(
                f"CHIRPS v2 native sampling contains missing values in {year}."
            )
        frame["cell_id"] = frame["cell_id"].astype(int)
        if frame.duplicated(["date", "cell_id"]).any():
            raise ValueError(
                f"CHIRPS v2 native sampling contains duplicates in {year}."
            )
        if len(frame) != len(expected_dates) * cells_per_day:
            raise ValueError(
                f"CHIRPS v2 native sampling returned {len(frame)} pixel-days "
                f"in {year}; expected "
                f"{len(expected_dates) * cells_per_day}."
            )
        actual_dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
        if not actual_dates.equals(expected_dates):
            raise ValueError(
                f"CHIRPS v2 native sampling dates are incomplete in {year}."
            )
        cell_counts = frame.groupby("date")["cell_id"].nunique()
        if not cell_counts.eq(cells_per_day).all():
            raise ValueError(
                f"CHIRPS v2 native sampling misses intersecting cells in {year}."
            )
        if (frame["precipitation"] < 0).any():
            raise ValueError(
                f"CHIRPS v2 native sampling has negative rainfall in {year}."
            )
        frames.append(frame)
        print(
            f"Sampled {len(expected_dates):,} days and {cells_per_day} "
            f"catchment cells for {year}."
        )

    samples = pd.concat(frames, ignore_index=True)
    samples = samples.merge(
        weights[["cell_id", "catchment_overlap_area_m2"]],
        on="cell_id",
        how="left",
        validate="many_to_one",
    )
    if samples["catchment_overlap_area_m2"].isna().any():
        raise ValueError("Native samples include an unknown catchment cell.")
    return samples.sort_values(["date", "cell_id"]).reset_index(drop=True)


def compare_reducers(
    samples: pd.DataFrame,
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    weighted = samples.assign(
        weighted_rainfall_mm=(
            samples["precipitation"]
            * samples["catchment_overlap_area_m2"]
        )
    )
    daily = weighted.groupby("date", as_index=False).agg(
        weighted_sum=("weighted_rainfall_mm", "sum"),
        valid_cells=("cell_id", "nunique"),
    )
    total_area = float(weights["catchment_overlap_area_m2"].sum())
    daily["v2_native_area_weighted_mm"] = daily["weighted_sum"] / total_area
    existing = pd.read_csv(V2_DAILY, parse_dates=["date"])
    expected_dates = pd.date_range(
        f"{START_YEAR}-01-01", f"{END_YEAR}-12-31", freq="D"
    )
    if existing["date"].duplicated().any():
        raise ValueError("The stored CHIRPS v2 catchment series has duplicates.")
    if not pd.DatetimeIndex(existing["date"]).equals(expected_dates):
        raise ValueError(
            "The stored CHIRPS v2 catchment series does not cover the "
            "expected complete period."
        )
    if not pd.DatetimeIndex(daily["date"]).equals(expected_dates):
        raise ValueError(
            "The sampled CHIRPS v2 native-grid series does not cover the "
            "expected complete period."
        )
    daily = daily.merge(
        existing.rename(
            columns={"precipitation_mm": "v2_earth_engine_mean_mm"}
        ),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    daily["difference_mm"] = (
        daily["v2_native_area_weighted_mm"]
        - daily["v2_earth_engine_mean_mm"]
    )
    daily["absolute_difference_mm"] = daily["difference_mm"].abs()
    if daily.isna().any().any():
        raise ValueError("Daily reducer comparison contains missing values.")

    daily["month_date"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    monthly = daily.groupby("month_date", as_index=False).agg(
        v2_native_area_weighted_mm=("v2_native_area_weighted_mm", "sum"),
        v2_earth_engine_mean_mm=("v2_earth_engine_mean_mm", "sum"),
        difference_mm=("difference_mm", "sum"),
        mean_absolute_daily_difference_mm=("absolute_difference_mm", "mean"),
        max_absolute_daily_difference_mm=("absolute_difference_mm", "max"),
        complete_days=("date", "count"),
    )
    monthly["difference_percent"] = (
        100
        * monthly["difference_mm"]
        / monthly["v2_earth_engine_mean_mm"].replace(0, np.nan)
    )
    daily["year"] = daily["date"].dt.year
    annual = daily.groupby("year", as_index=False).agg(
        v2_native_area_weighted_mm=("v2_native_area_weighted_mm", "sum"),
        v2_earth_engine_mean_mm=("v2_earth_engine_mean_mm", "sum"),
        difference_mm=("difference_mm", "sum"),
        mean_absolute_daily_difference_mm=("absolute_difference_mm", "mean"),
        max_absolute_daily_difference_mm=("absolute_difference_mm", "max"),
        complete_days=("date", "count"),
    )
    annual["difference_percent"] = (
        100
        * annual["difference_mm"]
        / annual["v2_earth_engine_mean_mm"].replace(0, np.nan)
    )
    difference = daily["difference_mm"]
    existing_total = float(daily["v2_earth_engine_mean_mm"].sum())
    weighted_total = float(daily["v2_native_area_weighted_mm"].sum())
    summary = {
        "start_date": daily["date"].min().strftime("%Y-%m-%d"),
        "end_date": daily["date"].max().strftime("%Y-%m-%d"),
        "days_compared": int(len(daily)),
        "native_grid_cells_sampled": int(len(weights)),
        "pixel_day_samples": int(len(samples)),
        "v2_native_area_weighted_total_mm": weighted_total,
        "v2_earth_engine_mean_total_mm": existing_total,
        "total_difference_mm": weighted_total - existing_total,
        "total_difference_percent": (
            100 * (weighted_total / existing_total - 1)
            if existing_total
            else None
        ),
        "daily_mean_difference_mm": float(difference.mean()),
        "daily_mean_absolute_difference_mm": float(
            daily["absolute_difference_mm"].mean()
        ),
        "daily_rmse_mm": float(np.sqrt(np.square(difference).mean())),
        "daily_max_absolute_difference_mm": float(
            daily["absolute_difference_mm"].max()
        ),
        "daily_correlation": float(
            daily["v2_native_area_weighted_mm"].corr(
                daily["v2_earth_engine_mean_mm"]
            )
        ),
        "monthly_mean_absolute_total_difference_mm": float(
            monthly["difference_mm"].abs().mean()
        ),
        "monthly_max_absolute_total_difference_mm": float(
            monthly["difference_mm"].abs().max()
        ),
        "annual_years_with_native_weighted_total_higher": int(
            annual["difference_mm"].gt(0).sum()
        ),
        "annual_years_with_native_weighted_total_lower": int(
            annual["difference_mm"].lt(0).sum()
        ),
        "annual_max_absolute_difference_percent": float(
            annual["difference_percent"].abs().max()
        ),
        "annual_difference_percent_quantiles": {
            str(round(float(q), 2)): float(
                annual["difference_percent"].quantile(q)
            )
            for q in (0, 0.25, 0.5, 0.75, 1)
        },
    }
    return daily, monthly, annual, summary


def main() -> None:
    if not CATCHMENT_GEOJSON.exists():
        raise FileNotFoundError(f"Catchment file not found: {CATCHMENT_GEOJSON}")
    if not V2_DAILY.exists():
        raise FileNotFoundError(f"CHIRPS v2 input not found: {V2_DAILY}")
    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Completed v2/v3 transition run not found: {OUTPUT_ROOT}"
        )
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Audit output already exists; preserve it and use a new run "
            f"directory if rerunning: {OUTPUT_DIR}"
        )

    with CATCHMENT_GEOJSON.open(encoding="utf-8") as source:
        geojson = json.load(source)
    features = geojson.get("features", [])
    if len(features) != 1:
        raise ValueError(
            f"Expected one saved catchment feature, found {len(features)}."
        )
    catchment = shape(features[0]["geometry"])
    if not catchment.is_valid or catchment.is_empty:
        raise ValueError("The saved catchment geometry is invalid or empty.")
    weights, catchment_area, area_coverage = get_intersecting_cells(catchment)

    initialize_earth_engine()
    samples = load_v2_native_grid_samples(weights)
    daily, monthly, annual, summary = compare_reducers(samples, weights)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    outputs = {
        "weights": OUTPUT_DIR / "catchment_grid_weights.csv",
        "cell_samples": OUTPUT_DIR / "v2_native_grid_cell_samples.csv",
        "daily": OUTPUT_DIR / "v2_spatial_reducer_daily_comparison.csv",
        "monthly": OUTPUT_DIR / "v2_spatial_reducer_monthly_comparison.csv",
        "annual": OUTPUT_DIR / "v2_spatial_reducer_annual_comparison.csv",
    }
    weights.to_csv(outputs["weights"], index=False)
    samples.to_csv(outputs["cell_samples"], index=False, date_format="%Y-%m-%d")
    daily.to_csv(outputs["daily"], index=False, date_format="%Y-%m-%d")
    monthly.to_csv(outputs["monthly"], index=False, date_format="%Y-%m-%d")
    annual.to_csv(outputs["annual"], index=False)
    output_manifest = {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in outputs.items()
    }

    projection = (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .filterDate("2025-01-01", "2025-01-02")
        .first()
        .select("precipitation")
        .projection()
        .getInfo()
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": PROJECT,
        "source_collection": CHIRPS_COLLECTION,
        "comparison_period": {
            "start_date": f"{START_YEAR}-01-01",
            "end_date": f"{END_YEAR}-12-31",
        },
        "native_grid": {
            "crs": projection["crs"],
            "transform": projection["transform"],
            "cell_size_degrees": CELL_SIZE_DEGREES,
            "sampling": (
                "Sample native precipitation values at each intersecting "
                "native-grid cell center using the image's native projection."
            ),
        },
        "catchment": {
            "file": str(CATCHMENT_GEOJSON),
            "sha256": sha256_file(CATCHMENT_GEOJSON),
            "area_m2_epsg_32736": catchment_area,
            "grid_overlap_area_coverage_fraction": float(area_coverage),
            "area_weight_projection": "EPSG:32736",
            "aggregation": (
                "Fractional intersection area of each native grid cell with "
                "the saved catchment polygon."
            ),
        },
        "existing_series": {
            "file": str(V2_DAILY),
            "sha256": sha256_file(V2_DAILY),
            "reducer": (
                "Earth Engine ee.Reducer.mean(), scale=5566, "
                "bestEffort=True."
            ),
        },
        "reference_product": {
            "transition_manifest": str(
                OUTPUT_ROOT / "transition_manifest.json"
            ),
            "transition_run_snapshot_id": json.loads(
                (OUTPUT_ROOT / "transition_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["icechunk_snapshot_id"],
            "comparison_method": (
                "Same CHIRPS v2 source pixels and saved catchment as the v3 "
                "comparison; only the reducer is changed to fractional "
                "native-grid cell-area weighting."
            ),
        },
        "results": summary,
        "outputs": output_manifest,
    }
    manifest_path = OUTPUT_DIR / "spatial_reducer_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved CHIRPS v2 spatial reducer audit to {OUTPUT_DIR}")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
