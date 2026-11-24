from __future__ import annotations

import ee
import numpy as np
import pandas as pd

from backend.data_ingestion import initialize_earth_engine, get_region


COLLECTIONS = {
    "chirps_v2": "UCSB-CHG/CHIRPS/DAILY",
    "chirps_v3_sat": "UCSB-CHC/CHIRPS/V3/DAILY_SAT",
    "chirps_v3_rnl": "UCSB-CHC/CHIRPS/V3/DAILY_RNL",
}

START_DATE = "2015-01-01"
END_DATE = "2026-11-01"  # Exclusive; includes October 2026.
SCALE = 5566


def fetch_monthly_collection(
    label: str,
    collection_id: str,
    region: ee.Geometry,
) -> pd.DataFrame:
    """Calculate catchment-mean monthly rainfall and image counts in Earth Engine."""
    print(f"\nRequesting {label}: {collection_id}", flush=True)

    collection = ee.ImageCollection(collection_id).select("precipitation")
    start = ee.Date(START_DATE)
    end = ee.Date(END_DATE)
    month_count = int(end.difference(start, "month").getInfo())

    def make_month_feature(offset):
        month_start = start.advance(offset, "month")
        month_end = month_start.advance(1, "month")
        month_images = collection.filterDate(month_start, month_end)
        image_count = month_images.size()

        monthly_total = ee.Algorithms.If(
            image_count.gt(0),
            month_images.sum().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=SCALE,
                bestEffort=True,
                maxPixels=1_000_000_000,
            ).get("precipitation"),
            None,
        )

        return ee.Feature(
            None,
            {
                "month": month_start.format("YYYY-MM"),
                "daily_observations": image_count,
                "monthly_rainfall_mm": monthly_total,
            },
        )

    features = ee.FeatureCollection(
        ee.List.sequence(0, month_count - 1).map(make_month_feature)
    )
    response = features.getInfo()
    rows = [feature["properties"] for feature in response["features"]]
    frame = pd.DataFrame(rows)

    frame["daily_observations"] = pd.to_numeric(
        frame["daily_observations"], errors="coerce"
    )
    frame["monthly_rainfall_mm"] = pd.to_numeric(
        frame["monthly_rainfall_mm"], errors="coerce"
    )
    frame["days_in_month"] = (
        pd.to_datetime(frame["month"] + "-01").dt.days_in_month
    )
    frame["complete_month"] = (
        frame["daily_observations"] == frame["days_in_month"]
    ) & frame["monthly_rainfall_mm"].notna()

    print(
        f"Returned {len(frame)} months; "
        f"{int(frame['complete_month'].sum())} complete months.",
        flush=True,
    )
    return frame.sort_values("month").reset_index(drop=True)


def compare(reference: pd.DataFrame, candidate: pd.DataFrame, label: str):
    """Compare only months with full daily coverage in both products."""
    paired = reference.merge(candidate, on="month", suffixes=("_v2", "_candidate"))
    paired = paired[
        paired["complete_month_v2"] & paired["complete_month_candidate"]
    ].copy()

    if paired.empty:
        print(f"\n{label}: no complete overlapping months.")
        return paired

    v2 = paired["monthly_rainfall_mm_v2"].astype(float)
    v3 = paired["monthly_rainfall_mm_candidate"].astype(float)
    difference = v3 - v2

    correlation = v2.corr(v3) if len(paired) > 1 else np.nan
    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    bias = float(difference.mean())
    reference_mean = float(v2.mean())
    percent_bias = 100 * bias / reference_mean if reference_mean else np.nan

    print(f"\n{'=' * 68}\n{label} versus CHIRPS v2\n{'=' * 68}")
    print(f"Complete overlapping months: {len(paired)}")
    print(f"Period: {paired['month'].iloc[0]} to {paired['month'].iloc[-1]}")
    print(f"Pearson correlation: {correlation:.4f}")
    print(f"MAE (mm/month): {mae:.4f}")
    print(f"RMSE (mm/month): {rmse:.4f}")
    print(f"Mean bias (mm/month; candidate - v2): {bias:.4f}")
    print(f"Percentage bias: {percent_bias:.2f}%")

    paired["calendar_month"] = pd.to_datetime(
        paired["month"] + "-01"
    ).dt.month
    seasonal = paired.groupby("calendar_month").agg(
        n=("month", "count"),
        v2_mean_mm=("monthly_rainfall_mm_v2", "mean"),
        candidate_mean_mm=("monthly_rainfall_mm_candidate", "mean"),
    )
    seasonal["bias_mm"] = (
        seasonal["candidate_mean_mm"] - seasonal["v2_mean_mm"]
    )
    print("\nCalendar-month means (complete paired months only):")
    print(seasonal.round(3).to_string())

    return paired


def main():
    initialize_earth_engine()
    region = get_region()

    print("=" * 68)
    print("WATERWATCH HISTORICAL CHIRPS MONTHLY VALIDATION")
    print("Mnjoli W60E catchment")
    print(f"Period: {START_DATE} to {END_DATE} (end exclusive)")
    print("Read-only: no production data, model, or API files are changed.")
    print("=" * 68)

    results = {}
    for label, collection_id in COLLECTIONS.items():
        results[label] = fetch_monthly_collection(
            label, collection_id, region
        )

    combined = results["chirps_v2"][
        ["month", "daily_observations", "monthly_rainfall_mm",
         "days_in_month", "complete_month"]
    ].rename(columns={
        "daily_observations": "v2_daily_observations",
        "monthly_rainfall_mm": "v2_rainfall_mm",
        "days_in_month": "v2_days_in_month",
        "complete_month": "v2_complete_month",
    })

    for label in ("chirps_v3_sat", "chirps_v3_rnl"):
        candidate = results[label][
            ["month", "daily_observations", "monthly_rainfall_mm",
             "days_in_month", "complete_month"]
        ].rename(columns={
            "daily_observations": f"{label}_daily_observations",
            "monthly_rainfall_mm": f"{label}_rainfall_mm",
            "days_in_month": f"{label}_days_in_month",
            "complete_month": f"{label}_complete_month",
        })
        combined = combined.merge(candidate, on="month", how="outer")

    output_path = "chirps_monthly_comparison_2015_2026.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    reference = results["chirps_v2"].rename(columns={
        "daily_observations": "daily_observations_v2",
        "monthly_rainfall_mm": "monthly_rainfall_mm_v2",
        "complete_month": "complete_month_v2",
    })

    for label in ("chirps_v3_sat", "chirps_v3_rnl"):
        candidate = results[label].rename(columns={
            "daily_observations": "daily_observations_candidate",
            "monthly_rainfall_mm": "monthly_rainfall_mm_candidate",
            "complete_month": "complete_month_candidate",
        })
        compare(reference, candidate, label.upper())

    print("\nLatest 15 monthly records:")
    print(combined.tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
