from __future__ import annotations

import pandas as pd

from backend.data_ingestion import (
    fetch_era5_daily,
    fetch_chirps_daily,
    latest_common_complete_month,
)


# ============================================================
# FROZEN TRAINING CLIMATOLOGY
# 2015–2021 only
# ============================================================

TRAINING_CLIMATOLOGY = {
    1: {"mean": -398.6579, "std": 158.9201},
    2: {"mean": -287.6825, "std": 160.1828},
    3: {"mean": -266.7649, "std": 155.2436},
    4: {"mean": -277.7504, "std": 106.6305},
    5: {"mean": -369.4874, "std": 43.9751},
    6: {"mean": -372.8801, "std": 31.3989},
    7: {"mean": -389.1788, "std": 43.4657},
    8: {"mean": -438.6049, "std": 25.1806},
    9: {"mean": -499.8706, "std": 24.0016},
    10: {"mean": -543.2090, "std": 34.4270},
    11: {"mean": -509.0128, "std": 103.6945},
    12: {"mean": -450.7918, "std": 146.5579},
}


# ============================================================
# LIVE HISTORY SETTINGS
# ============================================================

LIVE_HISTORY_START = pd.Timestamp("2026-01-01")


# ============================================================
# OBSERVED SWBA CALCULATION
# ============================================================

def calculate_observed_swba(
    monthly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the research target construction for
    observed future months.

    SWBA is based on the 3-month accumulated
    precipitation-minus-PET water balance and is
    standardized using the frozen 2015–2021
    calendar-month climatology.
    """

    print("LIVE HISTORY: calculating observed SWBA...")

    df = (
        monthly_df
        .copy()
        .sort_values("month_date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Monthly water balance
    # --------------------------------------------------------

    df["water_balance_mm"] = (
        df["precipitation_mm"]
        - df["pet_mm"]
    )

    # --------------------------------------------------------
    # 3-month accumulated water balance
    # --------------------------------------------------------

    df["water_balance_3month"] = (
        df["water_balance_mm"]
        .rolling(
            window=3,
            min_periods=3,
        )
        .sum()
    )

    # --------------------------------------------------------
    # Calendar month
    # --------------------------------------------------------

    df["calendar_month"] = (
        pd.to_datetime(
            df["month_date"]
        ).dt.month
    )

    # --------------------------------------------------------
    # Frozen training climatology
    # --------------------------------------------------------

    df["wb3_train_mean"] = (
        df["calendar_month"]
        .map(
            {
                month: values["mean"]
                for month, values
                in TRAINING_CLIMATOLOGY.items()
            }
        )
    )

    df["wb3_train_std"] = (
        df["calendar_month"]
        .map(
            {
                month: values["std"]
                for month, values
                in TRAINING_CLIMATOLOGY.items()
            }
        )
    )

    # --------------------------------------------------------
    # Standardized Water-Balance Anomaly
    # --------------------------------------------------------

    df["swba"] = (
        (
            df["water_balance_3month"]
            - df["wb3_train_mean"]
        )
        / df["wb3_train_std"]
    )

    print(
        "LIVE HISTORY: observed SWBA calculated."
    )

    return df


# ============================================================
# FETCH LIVE OBSERVED HISTORY
# ============================================================

def fetch_live_observed_history() -> pd.DataFrame:
    """
    Fetch observations from October 2025 through the
    latest complete month and calculate observed SWBA
    from January 2026 onward.

    October–December 2025 are fetched only to provide
    the antecedent months required by the 3-month
    rolling water-balance calculation.

    Diagnostic logging is included to identify which
    Earth Engine/data-processing stage is slow on Render.
    """

    print(
        "\n========================================"
    )
    print(
        "LIVE HISTORY: starting retrieval"
    )
    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Determine latest complete month
    # --------------------------------------------------------

    print(
        "LIVE HISTORY: determining latest complete month..."
    )

    latest_month = (
        latest_common_complete_month()
    )

    print(
        "LIVE HISTORY: latest complete month =",
        latest_month,
    )

    # --------------------------------------------------------
    # Check whether live history is available
    # --------------------------------------------------------

    if latest_month < LIVE_HISTORY_START:

        print(
            "LIVE HISTORY: latest month is before "
            "live-history start date."
        )

        return pd.DataFrame(
            columns=[
                "month_date",
                "water_balance_mm",
                "water_balance_3month",
                "wb3_train_mean",
                "wb3_train_std",
                "swba",
            ]
        )

    # --------------------------------------------------------
    # Fetch window
    # --------------------------------------------------------

    fetch_start = pd.Timestamp(
        "2025-10-01"
    )

    end_date = (
        latest_month
        + pd.offsets.MonthEnd(1)
        + pd.Timedelta(days=1)
    )

    print(
        "LIVE HISTORY: fetch window:",
        fetch_start.strftime("%Y-%m-%d"),
        "to",
        end_date.strftime("%Y-%m-%d"),
    )

    # --------------------------------------------------------
    # Fetch monthly water-balance inputs
    # --------------------------------------------------------
    #
    # The live-history calculation only needs monthly
    # precipitation and PET. Aggregate the daily Earth Engine
    # collections into monthly images before transferring data.
    # This keeps the observed-history methodology unchanged
    # while reducing the number of spatial reductions from
    # hundreds of daily images to the small number of months
    # actually required.

    print(
        "LIVE HISTORY: fetching monthly water-balance inputs..."
    )

    region = get_region()

    month_starts = pd.date_range(
        fetch_start,
        latest_month,
        freq="MS",
    )

    era5_monthly = []
    chirps_monthly = []

    for month_start in month_starts:

        month_end = (
            month_start
            + pd.offsets.MonthBegin(1)
        )

        start_str = month_start.strftime("%Y-%m-%d")
        end_str = month_end.strftime("%Y-%m-%d")

        era5_image = (
            ee.ImageCollection(
                "ECMWF/ERA5_LAND/DAILY_AGGR"
            )
            .filterDate(start_str, end_str)
            .select("potential_evaporation_sum")
            .sum()
        )

        era5_value = (
            era5_image
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=11132,
                bestEffort=True,
                maxPixels=1_000_000_000,
            )
            .get("potential_evaporation_sum")
        )

        chirps_image = (
            ee.ImageCollection(
                "UCSB-CHG/CHIRPS/DAILY"
            )
            .filterDate(start_str, end_str)
            .select("precipitation")
            .sum()
        )

        chirps_value = (
            chirps_image
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=5566,
                bestEffort=True,
                maxPixels=1_000_000_000,
            )
            .get("precipitation")
        )

        era5_monthly.append(
            {
                "month_date": month_start,
                "potential_evaporation_sum": (
                    era5_value.getInfo()
                    if era5_value is not None
                    else None
                ),
            }
        )

        chirps_monthly.append(
            {
                "month_date": month_start,
                "precipitation_mm": (
                    chirps_value.getInfo()
                    if chirps_value is not None
                    else None
                ),
            }
        )

    era5 = pd.DataFrame(era5_monthly)
    chirps = pd.DataFrame(chirps_monthly)

    if era5.empty or chirps.empty:
        raise RuntimeError(
            "No monthly ERA5-Land or CHIRPS data were returned."
        )

    daily = pd.merge(
        chirps[
            [
                "month_date",
                "precipitation_mm",
            ]
        ],
        era5[
            [
                "month_date",
                "potential_evaporation_sum",
            ]
        ],
        on="month_date",
        how="inner",
    )

    if daily.empty:
        raise RuntimeError(
            "No matching monthly dates between ERA5-Land and CHIRPS."
        )

    daily["pet_mm"] = (
        -daily["potential_evaporation_sum"] * 1000
    )

    monthly = daily[
        [
            "month_date",
            "precipitation_mm",
            "pet_mm",
        ]
    ].copy()

    print(
        "LIVE HISTORY: monthly data built."
    )

    # --------------------------------------------------------
    # Calculate SWBA
    # --------------------------------------------------------

    history = calculate_observed_swba(
        monthly
    )

    # --------------------------------------------------------
    # Select live period
    # --------------------------------------------------------

    print(
        "LIVE HISTORY: filtering 2026 observations..."
    )

    live_history = (
        history.loc[
            history["month_date"]
            >= LIVE_HISTORY_START,
            [
                "month_date",
                "water_balance_mm",
                "water_balance_3month",
                "wb3_train_mean",
                "wb3_train_std",
                "swba",
            ],
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "LIVE HISTORY: filtered history contains",
        len(live_history),
        "rows."
    )

    if not live_history.empty:

        print(
            "LIVE HISTORY: first observation =",
            live_history["month_date"].min(),
        )

        print(
            "LIVE HISTORY: last observation =",
            live_history["month_date"].max(),
        )

    print(
        "LIVE HISTORY: retrieval complete."
    )

    return live_history


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    from backend.data_ingestion import (
        initialize_earth_engine,
    )

    print(
        "\n========================================"
    )
    print(
        "LIVE OBSERVED SWBA HISTORY"
    )
    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Initialize Earth Engine
    # --------------------------------------------------------

    print(
        "LIVE HISTORY: initializing Earth Engine..."
    )

    initialize_earth_engine()

    print(
        "LIVE HISTORY: Earth Engine initialized."
    )

    # --------------------------------------------------------
    # Fetch history
    # --------------------------------------------------------

    history = fetch_live_observed_history()

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "\n========================================"
    )
    print(
        "RESULT"
    )
    print(
        "========================================"
    )

    if history.empty:

        print(
            "No live observed history available."
        )

    else:

        print(
            history.to_string(
                index=False
            )
        )

        print(
            "\nObservations:",
            len(history),
        )

        print(
            "Start:",
            history["month_date"].min(),
        )

        print(
            "End:",
            history["month_date"].max(),
        )

        print(
            "Latest observed SWBA:",
            round(
                float(
                    history.iloc[-1]["swba"]
                ),
                4,
            ),
        )