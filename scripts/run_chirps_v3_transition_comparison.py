from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
START_YEAR = 1981
DEFAULT_END_YEAR = 2025
BASELINE_START = 1981
BASELINE_END = 2010
V2_DAILY = ROOT / "data" / "external" / "expanded" / (
    "mnjoli_chirps_daily_1981_2025.csv"
)
HISTORICAL_MONTHLY = ROOT / "data" / "processed" / "expanded" / (
    "mnjoli_water_stress_monthly_1981_2025.csv"
)
HISTORICAL_SWBA = ROOT / "data" / "processed" / "expanded" / (
    "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)
CATCHMENT_GEOJSON = ROOT / "data" / "external" / (
    "mnjoli_w60e_catchment.geojson"
)
ICECHUNK_BUCKET = "dynamical-ucsb-chc-chirps"
ICECHUNK_PREFIX = (
    "ucsb-chc-chirps-analysis-final/v0.1.0.icechunk"
)
DATASET_ID = "ucsb-chc-chirps-analysis-final"
SWBA_THRESHOLD = -1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coordinate_slice(
    coordinates: np.ndarray,
    lower: float,
    upper: float,
    padding: float,
) -> slice:
    if coordinates[0] < coordinates[-1]:
        return slice(lower - padding, upper + padding)
    return slice(upper + padding, lower - padding)


def load_catchment(end_year: int) -> tuple[pd.DataFrame, str]:
    import icechunk
    from pyproj import Transformer
    from shapely import transform as transform_geometry
    from shapely.geometry import box, shape
    import xarray as xr

    with CATCHMENT_GEOJSON.open(encoding="utf-8") as source:
        geojson = json.load(source)
    features = geojson.get("features", [])
    if len(features) != 1:
        raise ValueError(
            f"Expected one saved Mnjoli catchment feature, found {len(features)}."
        )
    catchment = shape(features[0]["geometry"])
    if not catchment.is_valid or catchment.is_empty:
        raise ValueError("The saved Mnjoli catchment geometry is invalid or empty.")

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32736", always_xy=True
    )

    def project_geometry(geometry: Any) -> Any:
        def project_coordinates(coordinates: np.ndarray) -> np.ndarray:
            x, y = transformer.transform(
                coordinates[:, 0], coordinates[:, 1]
            )
            return np.column_stack((x, y))

        return transform_geometry(geometry, project_coordinates)

    projected_catchment = project_geometry(catchment)
    storage = icechunk.s3_storage(
        bucket=ICECHUNK_BUCKET,
        prefix=ICECHUNK_PREFIX,
        region="us-west-2",
        anonymous=True,
    )
    repository = icechunk.Repository.open(storage)
    session = repository.readonly_session("main")
    snapshot_id = session.snapshot_id

    end_date = pd.Timestamp(f"{end_year}-12-31")
    start_date = pd.Timestamp(f"{START_YEAR}-01-01")
    daily_frames: list[pd.DataFrame] = []
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None
    weights: np.ndarray | None = None
    total_weight = 0.0

    with xr.open_zarr(session.store, consolidated=False, chunks=None) as dataset:
        if "precipitation_surface" not in dataset:
            raise ValueError(
                f"Dataset {DATASET_ID} lacks precipitation_surface."
            )
        variable = dataset["precipitation_surface"]
        if variable.attrs.get("units") != "kg m-2 s-1":
            raise ValueError(
                "Unexpected CHIRPS v3 units: "
                f"{variable.attrs.get('units')!r}."
            )

        available_start = pd.Timestamp(dataset.time.values[0])
        available_end = pd.Timestamp(dataset.time.values[-1])
        if available_start > start_date or available_end < end_date:
            raise ValueError(
                "The mirror does not cover the requested complete period "
                f"{start_date:%Y-%m-%d} through {end_date:%Y-%m-%d}; it covers "
                f"{available_start:%Y-%m-%d} through {available_end:%Y-%m-%d}."
            )

        latitude_values = np.asarray(dataset.latitude.values)
        longitude_values = np.asarray(dataset.longitude.values)
        latitude_step = float(np.median(np.abs(np.diff(latitude_values))))
        longitude_step = float(np.median(np.abs(np.diff(longitude_values))))
        if not np.isclose(latitude_step, longitude_step, atol=1e-8):
            raise ValueError(
                "CHIRPS v3 latitude and longitude grid spacing differ."
            )
        if not np.isclose(latitude_step, 0.05, atol=1e-8):
            raise ValueError(
                f"Expected 0.05-degree CHIRPS v3 grid, got {latitude_step}."
            )
        half_cell = latitude_step / 2
        lat_slice = coordinate_slice(
            latitude_values,
            catchment.bounds[1],
            catchment.bounds[3],
            half_cell + 1e-8,
        )
        lon_slice = coordinate_slice(
            longitude_values,
            catchment.bounds[0],
            catchment.bounds[2],
            half_cell + 1e-8,
        )

        for year in range(START_YEAR, end_year + 1):
            year_data = variable.sel(
                time=slice(f"{year}-01-01", f"{year}-12-31"),
                latitude=lat_slice,
                longitude=lon_slice,
            ).compute()
            year_latitudes = np.asarray(year_data.latitude.values)
            year_longitudes = np.asarray(year_data.longitude.values)
            if not year_latitudes.size or not year_longitudes.size:
                raise ValueError(
                    f"No CHIRPS v3 grid cells intersect the catchment in {year}."
                )

            if latitudes is None:
                latitudes = year_latitudes
                longitudes = year_longitudes
                cell_weights: list[list[float]] = []
                for latitude in latitudes:
                    row: list[float] = []
                    for longitude in longitudes:
                        cell = project_geometry(
                            box(
                                longitude - half_cell,
                                latitude - half_cell,
                                longitude + half_cell,
                                latitude + half_cell,
                            ),
                        )
                        row.append(
                            projected_catchment.intersection(cell).area
                        )
                    cell_weights.append(row)
                weights = np.asarray(cell_weights, dtype=float)
                total_weight = float(weights.sum())
                if total_weight <= 0:
                    raise ValueError("Catchment/grid intersection has zero area.")
                area_coverage = total_weight / projected_catchment.area
                if not np.isclose(area_coverage, 1.0, rtol=0, atol=1e-6):
                    raise ValueError(
                        "Selected CHIRPS grid cells do not cover the catchment "
                        f"(area ratio {area_coverage:.9f})."
                    )
            elif not (
                np.array_equal(year_latitudes, latitudes)
                and np.array_equal(year_longitudes, longitudes)
            ):
                raise ValueError(
                    "CHIRPS v3 grid coordinates changed within the overlap period."
                )

            year_dates = pd.DatetimeIndex(year_data.time.values)
            expected_dates = pd.date_range(
                f"{year}-01-01", f"{year}-12-31", freq="D"
            )
            if not year_dates.equals(expected_dates):
                raise ValueError(
                    f"CHIRPS v3 daily coverage is incomplete or duplicated in {year}."
                )

            values = np.asarray(year_data.values, dtype=float)
            if np.isfinite(values).any() and np.nanmin(values) < 0:
                raise ValueError(
                    f"CHIRPS v3 contains negative precipitation in {year}."
                )
            valid = np.isfinite(values)
            valid_area = (valid * weights[None, :, :]).sum(axis=(1, 2))
            valid_fraction = valid_area / total_weight
            if np.any(valid_fraction < 0.999):
                bad_date = year_dates[int(np.argmin(valid_fraction))]
                raise ValueError(
                    "CHIRPS v3 has incomplete catchment coverage on "
                    f"{bad_date:%Y-%m-%d} "
                    f"({valid_fraction.min():.3%} area valid)."
                )
            safe_values = np.where(valid, values, 0.0)
            daily_rate = (
                (safe_values * weights[None, :, :]).sum(axis=(1, 2))
                / valid_area
            )
            daily_frames.append(
                pd.DataFrame(
                    {
                        "date": year_dates,
                        "precipitation_mm": daily_rate * 86400,
                        "valid_area_fraction": valid_fraction,
                    }
                )
            )
            print(f"Extracted CHIRPS v3 catchment data for {year}.")

    daily = pd.concat(daily_frames, ignore_index=True)
    expected_period = pd.date_range(start_date, end_date, freq="D")
    if not pd.DatetimeIndex(daily["date"]).equals(expected_period):
        raise ValueError("The extracted CHIRPS v3 dates are not continuous.")
    return daily, snapshot_id


def monthly_from_daily(daily: pd.DataFrame, source_name: str) -> pd.DataFrame:
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise")
    if work["date"].duplicated().any():
        raise ValueError(f"{source_name} contains duplicate daily dates.")
    if work["precipitation_mm"].isna().any():
        raise ValueError(f"{source_name} contains missing daily precipitation.")
    work["month_date"] = work["date"].dt.to_period("M").dt.to_timestamp()
    monthly = work.groupby("month_date", as_index=False).agg(
        precipitation_mm=("precipitation_mm", "sum"),
        valid_days=("date", "count"),
    )
    expected_days = monthly["month_date"].dt.days_in_month
    if not np.array_equal(monthly["valid_days"].to_numpy(), expected_days.to_numpy()):
        raise ValueError(f"{source_name} has incomplete monthly daily coverage.")
    return monthly


def calculate_swba(
    monthly: pd.DataFrame,
    precipitation_column: str,
    reference: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    balance = monthly[precipitation_column] - monthly["pet_mm"]
    balance_3month = balance.rolling(3, min_periods=3).sum()
    if reference is None:
        baseline = monthly["month_date"].dt.year.between(
            BASELINE_START, BASELINE_END
        )
        climatology = (
            pd.DataFrame(
                {
                    "month": monthly["month_date"].dt.month,
                    "balance": balance_3month,
                    "year": monthly["month_date"].dt.year,
                }
            )
            .loc[baseline]
            .groupby("month")["balance"]
            .agg(mean="mean", std="std")
        )
        if climatology.index.size != 12 or climatology.isna().any().any():
            raise ValueError("Could not calculate the full SWBA climatology.")
    else:
        climatology = reference

    month_numbers = monthly["month_date"].dt.month
    means = month_numbers.map(climatology["mean"])
    standard_deviations = month_numbers.map(climatology["std"])
    swba = (balance_3month - means) / standard_deviations
    return swba, climatology


def onset_dates(swba: pd.Series, dates: pd.Series) -> set[str]:
    onset = swba.le(SWBA_THRESHOLD) & swba.shift(1).gt(SWBA_THRESHOLD)
    return set(dates.loc[onset].dt.strftime("%Y-%m"))


def correlation(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique() < 2 or right.nunique() < 2:
        return None
    return float(left.corr(right))


def compare_monthly(
    v2_daily_path: Path,
    v3_daily: pd.DataFrame,
    end_year: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    v2_daily = pd.read_csv(v2_daily_path, parse_dates=["date"])
    expected_dates = pd.date_range(
        f"{START_YEAR}-01-01", f"{end_year}-12-31", freq="D"
    )
    for source_name, source_daily in (
        ("CHIRPS v2", v2_daily),
        ("CHIRPS v3", v3_daily),
    ):
        actual_dates = pd.DatetimeIndex(
            pd.to_datetime(source_daily["date"], errors="raise")
        )
        if not actual_dates.equals(expected_dates):
            raise ValueError(
                f"{source_name} daily dates do not exactly cover "
                f"{expected_dates[0]:%Y-%m-%d} through "
                f"{expected_dates[-1]:%Y-%m-%d}."
            )
    v2_monthly = monthly_from_daily(v2_daily, "CHIRPS v2")
    v3_monthly = monthly_from_daily(v3_daily, "CHIRPS v3")
    history = pd.read_csv(HISTORICAL_MONTHLY, parse_dates=["month_date"])
    history_swba = pd.read_csv(HISTORICAL_SWBA, parse_dates=["month_date"])

    monthly = (
        history.loc[history["month_date"].dt.year.between(START_YEAR, end_year)]
        .merge(
            v2_monthly[["month_date", "precipitation_mm"]].rename(
                columns={"precipitation_mm": "precipitation_v2_mm"}
            ),
            on="month_date",
            how="outer",
            validate="one_to_one",
        )
        .merge(
            v3_monthly[["month_date", "precipitation_mm"]].rename(
                columns={"precipitation_mm": "precipitation_v3_mm"}
            ),
            on="month_date",
            how="outer",
            validate="one_to_one",
        )
        .sort_values("month_date")
        .reset_index(drop=True)
    )
    expected_months = pd.date_range(
        f"{START_YEAR}-01-01", f"{end_year}-12-01", freq="MS"
    )
    if not pd.DatetimeIndex(monthly["month_date"]).equals(expected_months):
        raise ValueError("The v2, v3, and historical monthly ranges do not align.")
    for column in ("precipitation_v2_mm", "precipitation_v3_mm", "pet_mm"):
        if monthly[column].isna().any():
            raise ValueError(f"Monthly comparison has missing values in {column}.")

    original_precipitation_error = float(
        (
            monthly["precipitation_v2_mm"] - monthly["precipitation_mm"]
        ).abs().max()
    )
    if original_precipitation_error > 1e-6:
        raise ValueError(
            "The sum of the v2 daily series does not reproduce the historical "
            f"monthly CHIRPS totals (maximum difference "
            f"{original_precipitation_error:.9g} mm)."
        )

    monthly["precipitation_difference_mm"] = (
        monthly["precipitation_v3_mm"] - monthly["precipitation_v2_mm"]
    )
    monthly["water_balance_v2_mm"] = (
        monthly["precipitation_v2_mm"] - monthly["pet_mm"]
    )
    monthly["water_balance_v3_mm"] = (
        monthly["precipitation_v3_mm"] - monthly["pet_mm"]
    )
    monthly["water_balance_3month_v2_mm"] = (
        monthly["water_balance_v2_mm"].rolling(3, min_periods=3).sum()
    )
    monthly["water_balance_3month_v3_mm"] = (
        monthly["water_balance_v3_mm"].rolling(3, min_periods=3).sum()
    )

    monthly["swba_v2"], v2_reference = calculate_swba(
        monthly, "precipitation_v2_mm"
    )
    monthly["swba_v3_own_1981_2010"], _ = calculate_swba(
        monthly, "precipitation_v3_mm"
    )
    monthly["swba_v3_using_v2_reference"], _ = calculate_swba(
        monthly, "precipitation_v3_mm", v2_reference
    )
    monthly["drought_v2"] = monthly["swba_v2"].le(SWBA_THRESHOLD)
    monthly["drought_v3_own_1981_2010"] = monthly[
        "swba_v3_own_1981_2010"
    ].le(SWBA_THRESHOLD)
    monthly["drought_v3_using_v2_reference"] = monthly[
        "swba_v3_using_v2_reference"
    ].le(SWBA_THRESHOLD)

    stored_swba = history_swba[["month_date", "swba_3month"]]
    monthly = monthly.merge(
        stored_swba.rename(columns={"swba_3month": "stored_swba_v2"}),
        on="month_date",
        how="left",
        validate="one_to_one",
    )
    valid_v2_swba = monthly["swba_v2"].notna()
    v2_swba_error = float(
        (
            monthly.loc[valid_v2_swba, "swba_v2"]
            - monthly.loc[valid_v2_swba, "stored_swba_v2"]
        ).abs().max()
    )
    if v2_swba_error > 1e-8:
        raise ValueError(
            "Reconstructed v2 SWBA does not match the stored v2 index "
            f"(maximum difference {v2_swba_error:.9g})."
        )

    v2_onsets = onset_dates(monthly["swba_v2"], monthly["month_date"])
    v3_fixed_onsets = onset_dates(
        monthly["swba_v3_using_v2_reference"], monthly["month_date"]
    )
    v3_rebased_onsets = onset_dates(
        monthly["swba_v3_own_1981_2010"], monthly["month_date"]
    )
    compared = monthly["swba_v2"].notna()
    v2_labels = monthly.loc[compared, "drought_v2"]
    fixed_labels = monthly.loc[
        compared, "drought_v3_using_v2_reference"
    ]
    rebased_labels = monthly.loc[
        compared, "drought_v3_own_1981_2010"
    ]

    valid_precip = monthly["precipitation_v2_mm"].notna()
    v2_total = float(monthly.loc[valid_precip, "precipitation_v2_mm"].sum())
    v3_total = float(monthly.loc[valid_precip, "precipitation_v3_mm"].sum())
    difference = monthly["precipitation_difference_mm"]
    summary: dict[str, Any] = {
        "overlap_start_month": monthly["month_date"].min().strftime("%Y-%m"),
        "overlap_end_month": monthly["month_date"].max().strftime("%Y-%m"),
        "complete_months": int(len(monthly)),
        "v2_precipitation_total_mm": v2_total,
        "v3_precipitation_total_mm": v3_total,
        "precipitation_total_difference_mm": v3_total - v2_total,
        "precipitation_total_difference_percent": (
            100 * (v3_total - v2_total) / v2_total if v2_total else None
        ),
        "monthly_precipitation_bias_mm": float(difference.mean()),
        "monthly_precipitation_mae_mm": float(difference.abs().mean()),
        "monthly_precipitation_rmse_mm": float(
            np.sqrt(np.square(difference).mean())
        ),
        "monthly_precipitation_correlation": correlation(
            monthly["precipitation_v2_mm"],
            monthly["precipitation_v3_mm"],
        ),
        "monthly_precipitation_max_absolute_difference_mm": float(
            difference.abs().max()
        ),
        "stored_v2_monthly_precipitation_max_absolute_error_mm": (
            original_precipitation_error
        ),
        "stored_v2_swba_max_absolute_error": v2_swba_error,
        "swba_reference_period": f"{BASELINE_START}-{BASELINE_END}",
        "valid_swba_months": int(compared.sum()),
        "drought_months_v2": int(v2_labels.sum()),
        "drought_months_v3_using_v2_reference": int(fixed_labels.sum()),
        "drought_months_v3_own_1981_2010_reference": int(rebased_labels.sum()),
        "drought_label_changes_using_v2_reference": int(
            (v2_labels != fixed_labels).sum()
        ),
        "drought_label_changes_after_v3_rebaselining": int(
            (v2_labels != rebased_labels).sum()
        ),
        "onset_months_v2": sorted(v2_onsets),
        "onset_months_v3_using_v2_reference": sorted(v3_fixed_onsets),
        "onset_months_v3_own_1981_2010_reference": sorted(v3_rebased_onsets),
        "onset_events_v2": len(v2_onsets),
        "onset_events_v3_using_v2_reference": len(v3_fixed_onsets),
        "onset_events_v3_own_1981_2010_reference": len(v3_rebased_onsets),
        "onsets_removed_using_v2_reference": sorted(v2_onsets - v3_fixed_onsets),
        "onsets_added_using_v2_reference": sorted(v3_fixed_onsets - v2_onsets),
        "onsets_removed_after_v3_rebaselining": sorted(
            v2_onsets - v3_rebased_onsets
        ),
        "onsets_added_after_v3_rebaselining": sorted(
            v3_rebased_onsets - v2_onsets
        ),
    }

    monthly["calendar_month"] = monthly["month_date"].dt.month
    calendar_summary = monthly.groupby("calendar_month").agg(
        complete_months=("month_date", "count"),
        v2_mean_precipitation_mm=("precipitation_v2_mm", "mean"),
        v3_mean_precipitation_mm=("precipitation_v3_mm", "mean"),
        mean_difference_mm=("precipitation_difference_mm", "mean"),
        median_difference_mm=("precipitation_difference_mm", "median"),
        mean_absolute_difference_mm=(
            "precipitation_difference_mm",
            lambda values: values.abs().mean(),
        ),
    ).reset_index()
    monthly["year"] = monthly["month_date"].dt.year
    annual_summary = monthly.groupby("year").agg(
        v2_precipitation_total_mm=("precipitation_v2_mm", "sum"),
        v3_precipitation_total_mm=("precipitation_v3_mm", "sum"),
        v2_mean_3month_swba=("swba_v2", "mean"),
        v3_mean_3month_swba_using_v2_reference=(
            "swba_v3_using_v2_reference",
            "mean",
        ),
        v3_mean_3month_swba_own_reference=(
            "swba_v3_own_1981_2010",
            "mean",
        ),
    ).reset_index()
    annual_summary["precipitation_difference_mm"] = (
        annual_summary["v3_precipitation_total_mm"]
        - annual_summary["v2_precipitation_total_mm"]
    )
    annual_summary["precipitation_difference_percent"] = (
        100
        * annual_summary["precipitation_difference_mm"]
        / annual_summary["v2_precipitation_total_mm"].replace(0, np.nan)
    )
    return monthly, summary, calendar_summary, annual_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare catchment-aggregated CHIRPS v2 and final CHIRPS v3 "
            "over their common historical period."
        )
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help=f"Last complete overlap year (default: {DEFAULT_END_YEAR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to a timestamped versioned run folder.",
    )
    return parser.parse_args()


def main() -> None:
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "The CHIRPS v3 mirror uses Icechunk v2, which requires Python 3.12 "
            "or newer. Run this script with a Python 3.12+ environment and "
            "install requirements-chirps-v3-transition.txt."
        )

    args = parse_args()
    if args.end_year < BASELINE_END or args.end_year > DEFAULT_END_YEAR:
        raise ValueError(
            f"--end-year must be between {BASELINE_END} and {DEFAULT_END_YEAR}."
        )

    output_dir = args.output_dir or (
        ROOT
        / "outputs"
        / "expanded"
        / "chirps_v3_transition"
        / (
            f"run_through_{args.end_year}_"
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        )
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new one: {output_dir}"
        )

    daily, snapshot_id = load_catchment(args.end_year)
    monthly, comparison, calendar_summary, annual_summary = compare_monthly(
        V2_DAILY, daily, args.end_year
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    daily_path = output_dir / "v3_daily_catchment.csv"
    daily.to_csv(daily_path, index=False, date_format="%Y-%m-%d")
    monthly_path = output_dir / "v2_v3_monthly_comparison.csv"
    calendar_path = output_dir / "calendar_month_summary.csv"
    annual_path = output_dir / "annual_summary.csv"
    monthly.to_csv(monthly_path, index=False, date_format="%Y-%m-%d")
    calendar_summary.to_csv(calendar_path, index=False)
    annual_summary.to_csv(annual_path, index=False)

    geometry = json.loads(CATCHMENT_GEOJSON.read_text(encoding="utf-8"))
    feature = geometry["features"][0]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_ID,
        "product": "CHIRPS v3 final/rnl",
        "mirror": (
            "https://dynamical.org/catalog/ucsb-chc-chirps-analysis-final/"
        ),
        "upstream_source": (
            "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/"
        ),
        "license": "CC BY 4.0 for the dynamical.org processed dataset",
        "citation": (
            "UCSB Climate Hazards Center CHIRPS version 3.0 data processed by "
            "dynamical.org; Funk et al., Scientific Data 13, 718 (2026), "
            "https://doi.org/10.15780/G2JQ0P."
        ),
        "icechunk_bucket": ICECHUNK_BUCKET,
        "icechunk_prefix": ICECHUNK_PREFIX,
        "icechunk_snapshot_id": snapshot_id,
        "requested_start_date": f"{START_YEAR}-01-01",
        "requested_end_date": f"{args.end_year}-12-31",
        "catchment": {
            "file": str(CATCHMENT_GEOJSON),
            "sha256": sha256_file(CATCHMENT_GEOJSON),
            "hybas_id": feature.get("properties", {}).get("HYBAS_ID"),
            "projection_for_area_weights": "EPSG:32736",
            "aggregation": (
                "Fractional polygon/cell intersection area weights on the "
                "native 0.05-degree grid; daily weighted average."
            ),
        },
        "v2_source": {
            "collection": "UCSB-CHG/CHIRPS/DAILY",
            "daily_file": str(V2_DAILY),
            "daily_file_sha256": sha256_file(V2_DAILY),
            "monthly_file": str(HISTORICAL_MONTHLY),
            "monthly_file_sha256": sha256_file(HISTORICAL_MONTHLY),
            "swba_file": str(HISTORICAL_SWBA),
            "swba_file_sha256": sha256_file(HISTORICAL_SWBA),
            "spatial_reducer": (
                "Earth Engine mean, scale 5566 m, bestEffort=True, on the "
                "saved Mnjoli HydroSHEDS catchment."
            ),
        },
        "v3_source_units": "kg m-2 s-1, converted to mm/day by multiplying 86400",
        "v3_source_release_timing": (
            "The final dataset catalog states monthly release, usually two to "
            "three weeks after month end."
        ),
        "comparison_method": {
            "pet": "Held at the existing ERA5-Land monthly series for both versions.",
            "swba": (
                "Three-month precipitation-minus-PET balance standardized by "
                "calendar month using 1981-2010 sample mean and standard deviation."
            ),
            "v3_swba_variants": [
                "v3 precipitation with the v2 1981-2010 reference statistics",
                "v3 precipitation with its own 1981-2010 reference statistics",
            ],
            "drought_threshold": SWBA_THRESHOLD,
            "onset": (
                "First SWBA <= -1 month following a valid month with SWBA > -1."
            ),
        },
        "software": {
            "python": sys.version,
            "icechunk": __import__("icechunk").__version__,
            "xarray": __import__("xarray").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "outputs": {
            "v3_daily_catchment_csv": {
                "path": str(daily_path),
                "sha256": sha256_file(daily_path),
            },
            "monthly_comparison_csv": {
                "path": str(monthly_path),
                "sha256": sha256_file(monthly_path),
            },
            "calendar_month_summary_csv": {
                "path": str(calendar_path),
                "sha256": sha256_file(calendar_path),
            },
            "annual_summary_csv": {
                "path": str(annual_path),
                "sha256": sha256_file(annual_path),
            },
        },
        "results": comparison,
        "interpretation_caveat": (
            "The v2 series was reduced in Earth Engine at scale 5566 m with "
            "bestEffort=True; v3 uses fractional native-grid cell overlap in "
            "EPSG:32736. The methods use the same catchment, but are not "
            "identical pixel grids/reducers. Attribute small differences "
            "cautiously; this analysis does not alter the live model."
        ),
    }
    manifest_path = output_dir / "transition_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved CHIRPS v3 transition results to {output_dir}")
    print(json.dumps(comparison, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
