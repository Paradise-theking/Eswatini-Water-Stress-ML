from pathlib import Path

import pandas as pd

from data_ingestion import (
    initialize_earth_engine,
    fetch_era5_daily,
    fetch_chirps_daily,
    transform_daily_data,
    aggregate_monthly,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mnjoli_water_stress_ml_dataset.csv"
)


FEATURES_TO_COMPARE = [
    "precipitation_mm",
    "pet_mm",
    "temperature_max_c",
    "soil_moisture_layer1",
    "soil_moisture_layer2",
    "runoff_mm",
    "surface_runoff_mm",
    "solar_radiation",
]


def main():

    initialize_earth_engine()

    print("\n========================================")
    print("INGESTION VALIDATION — DECEMBER 2025")
    print("========================================")

    # -----------------------------------------
    # Existing research dataset
    # -----------------------------------------

    historical = pd.read_csv(
        HISTORICAL_PATH
    )

    historical["month_date"] = pd.to_datetime(
        historical["month_date"]
    )

    old_row = historical.loc[
        historical["month_date"]
        == pd.Timestamp("2025-12-01")
    ]

    if old_row.empty:
        raise RuntimeError(
            "December 2025 was not found in the historical dataset."
        )

    old_row = old_row.iloc[0]

    # -----------------------------------------
    # Fresh Earth Engine ingestion
    # -----------------------------------------

    print("\nFetching December 2025 again from Earth Engine...")

    era5 = fetch_era5_daily(
        "2025-12-01",
        "2026-01-01",
    )

    chirps = fetch_chirps_daily(
        "2025-12-01",
        "2026-01-01",
    )

    daily = transform_daily_data(
        era5,
        chirps,
    )

    fresh_monthly = aggregate_monthly(
        daily
    )

    if fresh_monthly.empty:
        raise RuntimeError(
            "Fresh December 2025 ingestion produced no monthly row."
        )

    new_row = fresh_monthly.iloc[0]

    # -----------------------------------------
    # Compare
    # -----------------------------------------

    comparisons = []

    for feature in FEATURES_TO_COMPARE:

        historical_value = float(
            old_row[feature]
        )

        fresh_value = float(
            new_row[feature]
        )

        difference = (
            fresh_value
            - historical_value
        )

        if historical_value != 0:
            percent_difference = (
                difference
                / abs(historical_value)
                * 100
            )
        else:
            percent_difference = float("nan")

        comparisons.append(
            {
                "feature": feature,
                "historical": historical_value,
                "fresh": fresh_value,
                "difference": difference,
                "percent_difference": percent_difference,
            }
        )

    comparison_df = pd.DataFrame(
        comparisons
    )

    print("\n========================================")
    print("COMPARISON")
    print("========================================")

    print(
        comparison_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()