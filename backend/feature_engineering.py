from __future__ import annotations

import pandas as pd


MODEL_FEATURES = [
    "soil_moisture_layer1_lag1",
    "precipitation_mm_lag1",
    "pet_mm",
    "soil_moisture_layer2_lag1",
    "precipitation_mm",
    "soil_moisture_layer2",
    "precipitation_3month",
    "soil_moisture_layer1",
    "pet_mm_lag1",
    "temperature_max_c",
    "surface_runoff_mm_lag1",
    "runoff_mm_lag1",
    "pet_3month",
    "solar_radiation",
    "solar_radiation_lag1",
]


def add_forecast_features(
    monthly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the lagged and rolling predictors required
    by the trained Ridge forecasting model.
    """

    df = monthly_df.copy()

    df = df.sort_values(
        "month_date"
    ).reset_index(drop=True)

    # -----------------------------------------
    # Lag-1 predictors
    # -----------------------------------------

    lag_variables = [
        "soil_moisture_layer1",
        "precipitation_mm",
        "soil_moisture_layer2",
        "pet_mm",
        "surface_runoff_mm",
        "runoff_mm",
        "solar_radiation",
    ]

    for variable in lag_variables:
        df[f"{variable}_lag1"] = (
            df[variable].shift(1)
        )

    # -----------------------------------------
    # 3-month accumulated variables
    # -----------------------------------------

    df["precipitation_3month"] = (
        df["precipitation_mm"]
        .rolling(
            window=3,
            min_periods=3,
        )
        .sum()
    )

    df["pet_3month"] = (
        df["pet_mm"]
        .rolling(
            window=3,
            min_periods=3,
        )
        .sum()
    )

    return df


def build_latest_model_input(
    monthly_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Build the final 15-feature input row for the latest
    available complete month.
    """

    engineered = add_forecast_features(
        monthly_df
    )

    latest = engineered.iloc[-1]

    missing = [
        feature
        for feature in MODEL_FEATURES
        if pd.isna(latest[feature])
    ]

    if missing:
        raise ValueError(
            "Unable to construct model input. "
            f"Missing features: {missing}"
        )

    model_input = pd.DataFrame(
        [
            {
                feature: float(
                    latest[feature]
                )
                for feature in MODEL_FEATURES
            }
        ],
        columns=MODEL_FEATURES,
    )

    observation_month = pd.Timestamp(
        latest["month_date"]
    )

    return (
        model_input,
        observation_month,
    )


def forecast_month_from_observation(
    observation_month: pd.Timestamp,
) -> pd.Timestamp:
    """
    The model performs one-month-ahead forecasting.
    """

    return (
        observation_month
        + pd.DateOffset(months=1)
    )


if __name__ == "__main__":

    from data_ingestion import (
        initialize_earth_engine,
        fetch_latest_monthly_data,
    )

    print(
        "\n========================================"
    )
    print(
        "LIVE FEATURE ENGINEERING TEST"
    )
    print(
        "========================================"
    )

    initialize_earth_engine()

    monthly = fetch_latest_monthly_data()

    model_input, observation_month = (
        build_latest_model_input(
            monthly
        )
    )

    forecast_month = (
        forecast_month_from_observation(
            observation_month
        )
    )

    print(
        "\nObservation month:",
        observation_month.strftime("%Y-%m"),
    )

    print(
        "Forecast month:",
        forecast_month.strftime("%Y-%m"),
    )

    print(
        "\n========================================"
    )
    print(
        "15 MODEL FEATURES"
    )
    print(
        "========================================"
    )

    print(
        model_input.T.to_string(
            header=False
        )
    )