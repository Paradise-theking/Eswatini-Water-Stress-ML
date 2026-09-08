from __future__ import annotations

import pandas as pd

from backend.data_ingestion import (
    fetch_era5_daily,
    fetch_chirps_daily,
    transform_daily_data,
    aggregate_monthly,
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


LIVE_HISTORY_START = pd.Timestamp("2026-01-01")


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

    df = (
        monthly_df
        .copy()
        .sort_values("month_date")
        .reset_index(drop=True)
    )

    # Monthly water balance
    df["water_balance_mm"] = (
        df["precipitation_mm"]
        - df["pet_mm"]
    )

    # 3-month accumulated water balance
    df["water_balance_3month"] = (
        df["water_balance_mm"]
        .rolling(
            window=3,
            min_periods=3,
        )
        .sum()
    )

    # Calendar month
    df["calendar_month"] = (
        pd.to_datetime(
            df["month_date"]
        ).dt.month
    )

    # Frozen training climatology
    df["wb3_train_mean"] = (
        df["calendar_month"]
        .apply(
            lambda month:
            TRAINING_CLIMATOLOGY[
                int(month)
            ]["mean"]
        )
    )

    df["wb3_train_std"] = (
        df["calendar_month"]
        .apply(
            lambda month:
            TRAINING_CLIMATOLOGY[
                int(month)
            ]["std"]
        )
    )

    # Standardized Water-Balance Anomaly
    df["swba"] = (
        (
            df["water_balance_3month"]
            - df["wb3_train_mean"]
        )
        / df["wb3_train_std"]
    )

    return df


def fetch_live_observed_history() -> pd.DataFrame:
    """
    Fetch observations from October 2025 through the
    latest complete month and calculate observed SWBA
    from January 2026 onward.

    October–December 2025 are fetched only to provide
    the antecedent months required by the 3-month
    rolling water-balance calculation.
    """

    latest_month = (
        latest_common_complete_month()
    )

    if latest_month < LIVE_HISTORY_START:
        return pd.DataFrame(
            columns=[
                "month_date",
                "swba",
            ]
        )

    fetch_start = pd.Timestamp(
        "2025-10-01"
    )

    end_date = (
        latest_month
        + pd.offsets.MonthEnd(1)
        + pd.Timedelta(days=1)
    )

    era5 = fetch_era5_daily(
        fetch_start.strftime(
            "%Y-%m-%d"
        ),
        end_date.strftime(
            "%Y-%m-%d"
        ),
    )

    chirps = fetch_chirps_daily(
        fetch_start.strftime(
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

    history = calculate_observed_swba(
        monthly
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

    return live_history


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

    initialize_earth_engine()

    history = fetch_live_observed_history()

    print(
        history.to_string(
            index=False
        )
    )

    print(
        "\nObservations:",
        len(history),
    )

    if not history.empty:
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