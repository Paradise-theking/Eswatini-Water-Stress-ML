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
END_DATE = "2026-11-01"  # Exclusive: includes 31 October 2026.
SCALE = 5566
OUTPUT_PATH = "chirps_daily_comparison_2015_2026.csv"


def fetch_daily_collection(
    label: str,
    collection_id: str,
    region: ee.Geometry,
) -> pd.DataFrame:
    """Extract daily catchment-mean rainfall, retrieving one year per request."""
    print(f"\nFetching {label}: {collection_id}", flush=True)

    collection = (
        ee.ImageCollection(collection_id)
        .select("precipitation")
        .filterDate(START_DATE, END_DATE)
    )

    rows = []

    for year in range(
        int(START_DATE[:4]),
        int(END_DATE[:4]) + 1,
    ):
        year_start = ee.Date.fromYMD(year, 1, 1)
        year_end = year_start.advance(1, "year")
        yearly = collection.filterDate(year_start, year_end)
        image_count = yearly.size()

        # ImageCollection.map() must return images. Convert to a server-side
        # list first so each mapped item can correctly be an ee.Feature.
        image_list = yearly.toList(image_count)
        indices = ee.List.sequence(0, image_count.subtract(1))

        def image_to_feature(index):
            image = ee.Image(image_list.get(index))

            rainfall = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=SCALE,
                bestEffort=True,
                maxPixels=1_000_000_000,
            ).get("precipitation")

            date = ee.Date(
                image.get("system:time_start")
            ).format("YYYY-MM-dd")

            return ee.Feature(
                None,
                {
                    "date": date,
                    "rainfall_mm": rainfall,
                },
            )

        features = ee.FeatureCollection(indices.map(image_to_feature))
        response = features.getInfo()
        year_rows = [
            feature["properties"]
            for feature in response["features"]
        ]
        rows.extend(year_rows)

        print(
            f"  {year}: {len(year_rows)} daily images returned",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No daily observations returned for {label}.")

    frame["date"] = pd.to_datetime(frame["date"])
    frame["rainfall_mm"] = pd.to_numeric(
        frame["rainfall_mm"], errors="coerce"
    )
    frame = (
        frame.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    print(
        f"Total images: {len(frame)}; "
        f"valid catchment rainfall values: {frame['rainfall_mm'].notna().sum()}",
        flush=True,
    )
    return frame


def compare(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Compare paired daily rainfall values and rainfall-occurrence categories."""
    paired = reference.merge(
        candidate,
        on="date",
        how="inner",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    paired = paired.dropna(
        subset=["rainfall_mm_reference", "rainfall_mm_candidate"]
    ).copy()

    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)

    if paired.empty:
        print("No paired valid daily observations.")
        return paired

    ref = paired["rainfall_mm_reference"].astype(float)
    cand = paired["rainfall_mm_candidate"].astype(float)
    delta = cand - ref

    wet_ref = ref >= 1.0
    wet_cand = cand >= 1.0
    heavy_ref = ref >= 20.0
    heavy_cand = cand >= 20.0
    dry_ref = ref < 1.0
    dry_cand = cand < 1.0

    print(f"Paired daily observations: {len(paired)}")
    print(
        f"Date range: {paired['date'].min():%Y-%m-%d} "
        f"to {paired['date'].max():%Y-%m-%d}"
    )
    print(f"Pearson correlation: {ref.corr(cand):.6f}")
    print(f"MAE: {np.mean(np.abs(delta)):.6f} mm/day")
    print(f"RMSE: {np.sqrt(np.mean(delta ** 2)):.6f} mm/day")
    print(f"Mean bias (candidate - reference): {delta.mean():+.6f} mm/day")
    print(f"Maximum absolute difference: {np.abs(delta).max():.6f} mm/day")

    print("\nRainfall occurrence (threshold: 1 mm/day):")
    print(
        f"Reference wet days: {wet_ref.sum()} "
        f"({wet_ref.mean() * 100:.2f}%)"
    )
    print(
        f"Candidate wet days: {wet_cand.sum()} "
        f"({wet_cand.mean() * 100:.2f}%)"
    )
    print(f"Both wet: {(wet_ref & wet_cand).sum()}")
    print(f"Both dry: {(dry_ref & dry_cand).sum()}")
    print(f"Reference wet / candidate dry: {(wet_ref & dry_cand).sum()}")
    print(f"Reference dry / candidate wet: {(dry_ref & wet_cand).sum()}")

    print("\nHeavy rainfall (threshold: 20 mm/day):")
    print(f"Reference heavy days: {heavy_ref.sum()}")
    print(f"Candidate heavy days: {heavy_cand.sum()}")
    print(f"Both heavy: {(heavy_ref & heavy_cand).sum()}")
    print(
        "Reference heavy / candidate not heavy: "
        f"{(heavy_ref & ~heavy_cand).sum()}"
    )
    print(
        "Reference not heavy / candidate heavy: "
        f"{(~heavy_ref & heavy_cand).sum()}"
    )

    paired["absolute_difference_mm"] = np.abs(delta)
    print("\nTen largest daily absolute differences:")
    print(
        paired.nlargest(10, "absolute_difference_mm")[
            [
                "date",
                "rainfall_mm_reference",
                "rainfall_mm_candidate",
                "absolute_difference_mm",
            ]
        ].to_string(index=False)
    )

    return paired


def main():
    initialize_earth_engine()
    region = get_region()

    print("=" * 72)
    print("WATERWATCH DAILY CHIRPS PRODUCT COMPARISON")
    print("Mnjoli W60E catchment")
    print(f"Period: {START_DATE} to {END_DATE} (end exclusive)")
    print("Read-only diagnostic: production data and model are not modified.")
    print("=" * 72)

    results = {
        label: fetch_daily_collection(label, collection_id, region)
        for label, collection_id in COLLECTIONS.items()
    }

    # Preserve the full date union so missing product-days remain visible.
    combined = None
    for label, frame in results.items():
        current = frame.rename(
            columns={"rainfall_mm": f"{label}_rainfall_mm"}
        )
        combined = (
            current
            if combined is None
            else combined.merge(current, on="date", how="outer")
        )

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved daily series: {OUTPUT_PATH}")

    expected_dates = pd.date_range(
        START_DATE,
        pd.Timestamp(END_DATE) - pd.Timedelta(days=1),
        freq="D",
    )
    print("\nDaily coverage audit:")
    print(f"Expected calendar days: {len(expected_dates)}")
    for label in COLLECTIONS:
        column = f"{label}_rainfall_mm"
        observed_dates = combined.loc[combined[column].notna(), "date"]
        missing_dates = expected_dates.difference(observed_dates)
        print(
            f"{label}: valid={len(observed_dates)}, "
            f"missing={len(missing_dates)}"
        )
        if len(missing_dates):
            print(
                "  First missing dates: "
                + ", ".join(str(date.date()) for date in missing_dates[:10])
            )

    compare(
        results["chirps_v2"],
        results["chirps_v3_sat"],
        "CHIRPS V3 SAT versus CHIRPS V2",
    )
    compare(
        results["chirps_v2"],
        results["chirps_v3_rnl"],
        "CHIRPS V3 RNL versus CHIRPS V2",
    )
    compare(
        results["chirps_v3_sat"],
        results["chirps_v3_rnl"],
        "CHIRPS V3 SAT versus CHIRPS V3 RNL",
    )

    print("\nCompleted. No production files were changed.")


if __name__ == "__main__":
    main()
