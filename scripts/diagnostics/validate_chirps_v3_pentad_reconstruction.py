from __future__ import annotations

import ee
import numpy as np
import pandas as pd

from backend.data_ingestion import initialize_earth_engine, get_region


PENTAD_COLLECTION = "UCSB-CHC/CHIRPS/V3/PENTAD"
DAILY_COLLECTIONS = {
    "chirps_v3_sat": "UCSB-CHC/CHIRPS/V3/DAILY_SAT",
    "chirps_v3_rnl": "UCSB-CHC/CHIRPS/V3/DAILY_RNL",
}

START_DATE = "2015-01-01"
END_DATE = "2026-11-01"  # Exclusive; includes the October 2026 pentad.
SCALE = 5566
OUTPUT_PATH = "chirps_v3_pentad_reconstruction_2015_2026.csv"


def fetch_year(
    year: int,
    region: ee.Geometry,
    pentads: ee.ImageCollection,
    daily_collections: dict[str, ee.ImageCollection],
) -> list[dict]:
    """Compare each official pentad with sums of its daily SAT and RNL images."""
    yearly_pentads = pentads.filter(ee.Filter.eq("year", year))
    image_count = yearly_pentads.size().getInfo()
    if not image_count:
        print(f"  {year}: no official pentad images", flush=True)
        return []

    def pentad_to_feature(image):
        image = ee.Image(image)
        year_value = ee.Number(image.get("year")).toInt()
        month_value = ee.Number(image.get("month")).toInt()
        pentad_value = ee.Number(image.get("pentad")).toInt()

        month_start = ee.Date.fromYMD(year_value, month_value, 1)
        start_day = pentad_value.subtract(1).multiply(5).add(1)
        pentad_start = ee.Date.fromYMD(
            year_value, month_value, start_day
        )
        pentad_end = ee.Date(
            ee.Algorithms.If(
                pentad_value.eq(6),
                month_start.advance(1, "month"),
                pentad_start.advance(5, "day"),
            )
        )
        expected_days = pentad_end.difference(pentad_start, "day")

        official_value = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=SCALE,
            bestEffort=True,
            maxPixels=1_000_000_000,
        ).get("precipitation")

        properties = {
            "year": year_value,
            "month": month_value,
            "pentad": pentad_value,
            "start_date": pentad_start.format("YYYY-MM-dd"),
            "end_date_exclusive": pentad_end.format("YYYY-MM-dd"),
            "expected_days": expected_days,
            "official_pentad_mm": official_value,
        }

        for label, daily_collection in daily_collections.items():
            daily_period = daily_collection.filterDate(
                pentad_start, pentad_end
            )
            daily_count = daily_period.size()
            daily_sum = ee.Algorithms.If(
                daily_count.gt(0),
                daily_period.sum().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=SCALE,
                    bestEffort=True,
                    maxPixels=1_000_000_000,
                ).get("precipitation"),
                None,
            )
            properties[f"{label}_daily_count"] = daily_count
            properties[f"{label}_daily_sum_mm"] = daily_sum

        return ee.Feature(None, properties)

    features = ee.FeatureCollection(yearly_pentads.map(pentad_to_feature))
    response = features.getInfo()
    rows = [feature["properties"] for feature in response["features"]]
    print(f"  {year}: {len(rows)} pentads returned", flush=True)
    return rows


def report_metrics(frame: pd.DataFrame, label: str, daily_column: str):
    """Report reconstruction errors only for complete daily pentads."""
    count_column = f"{label}_daily_count"
    paired = frame.dropna(
        subset=["official_pentad_mm", daily_column, count_column]
    ).copy()
    paired = paired[
        paired[count_column] == paired["expected_days"]
    ].copy()

    print("\n" + "=" * 72)
    print(f"{label.upper()} DAILY-TO-PENTAD RECONSTRUCTION")
    print("=" * 72)
    print(f"Complete pentads: {len(paired)}")

    if paired.empty:
        print("No complete pentads available for comparison.")
        return

    official = paired["official_pentad_mm"].astype(float)
    reconstructed = paired[daily_column].astype(float)
    difference = reconstructed - official

    correlation = official.corr(reconstructed)
    mae = np.mean(np.abs(difference))
    rmse = np.sqrt(np.mean(np.square(difference)))
    bias = difference.mean()
    max_abs = np.abs(difference).max()

    print(
        f"Period: {paired['start_date'].min()} to "
        f"{paired['end_date_exclusive'].max()} (end exclusive)"
    )
    print(f"Pearson correlation: {correlation:.10f}")
    print(f"MAE: {mae:.10f} mm/pentad")
    print(f"RMSE: {rmse:.10f} mm/pentad")
    print(f"Mean bias (daily sum - official): {bias:+.10f} mm/pentad")
    print(f"Maximum absolute difference: {max_abs:.10f} mm/pentad")

    paired["absolute_difference_mm"] = np.abs(difference)
    print("\nTen largest absolute reconstruction differences:")
    print(
        paired.nlargest(10, "absolute_difference_mm")[
            [
                "start_date",
                "end_date_exclusive",
                "official_pentad_mm",
                daily_column,
                "absolute_difference_mm",
            ]
        ].to_string(index=False)
    )


def main():
    initialize_earth_engine()
    region = get_region()

    print("=" * 72)
    print("WATERWATCH CHIRPS V3 PENTAD MASS-BALANCE VALIDATION")
    print("Mnjoli W60E catchment")
    print(f"Period: {START_DATE} to {END_DATE} (end exclusive)")
    print(f"Official reference: {PENTAD_COLLECTION}")
    print("Read-only: production data, features, model and API are not modified.")
    print("=" * 72)

    pentads = (
        ee.ImageCollection(PENTAD_COLLECTION)
        .select("precipitation")
        .filterDate(START_DATE, END_DATE)
    )
    daily_collections = {
        label: (
            ee.ImageCollection(collection_id)
            .select("precipitation")
            .filterDate(START_DATE, END_DATE)
        )
        for label, collection_id in DAILY_COLLECTIONS.items()
    }

    all_rows: list[dict] = []
    for year in range(
        int(START_DATE[:4]),
        int(END_DATE[:4]) + 1,
    ):
        all_rows.extend(
            fetch_year(year, region, pentads, daily_collections)
        )

    if not all_rows:
        raise RuntimeError("No CHIRPS v3 pentad observations were returned.")

    frame = pd.DataFrame(all_rows)
    numeric_columns = [
        "year",
        "month",
        "pentad",
        "expected_days",
        "official_pentad_mm",
    ]
    for label in DAILY_COLLECTIONS:
        numeric_columns.extend(
            [
                f"{label}_daily_count",
                f"{label}_daily_sum_mm",
            ]
        )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values(
        ["year", "month", "pentad"]
    ).reset_index(drop=True)

    for label in DAILY_COLLECTIONS:
        frame[f"{label}_difference_mm"] = (
            frame[f"{label}_daily_sum_mm"] - frame["official_pentad_mm"]
        )

    frame.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved pentad reconstruction series: {OUTPUT_PATH}")

    print("\nCoverage audit:")
    print(f"Official pentad records returned: {len(frame)}")
    print(
        "Official pentads with catchment values: "
        f"{frame['official_pentad_mm'].notna().sum()}"
    )
    for label in DAILY_COLLECTIONS:
        count_column = f"{label}_daily_count"
        complete = (
            frame[count_column] == frame["expected_days"]
        ) & frame[f"{label}_daily_sum_mm"].notna()
        print(
            f"{label}: complete daily pentads={int(complete.sum())}; "
            f"incomplete or missing={int((~complete).sum())}"
        )

    report_metrics(
        frame,
        "chirps_v3_sat",
        "chirps_v3_sat_daily_sum_mm",
    )
    report_metrics(
        frame,
        "chirps_v3_rnl",
        "chirps_v3_rnl_daily_sum_mm",
    )

    print("\nCompleted. No production files were changed.")


if __name__ == "__main__":
    main()
