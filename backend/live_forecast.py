from __future__ import annotations

from pathlib import Path

import joblib

from backend.data_ingestion import (
    fetch_latest_monthly_data,
    initialize_earth_engine,
    latest_common_complete_month,
)

from backend.feature_engineering import (
    MODEL_FEATURES,
    build_latest_model_input,
    forecast_month_from_observation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "water_stress_ridge.joblib"
)


def classify_water_stress(
    prediction: float,
) -> tuple[str, str]:
    """
    Convert predicted Water Stress Index into
    a human-readable category.

    Thresholds match the frontend dashboard logic.
    """

    if prediction <= -2.0:
        return (
            "Extreme Water Stress",
            "Exceptionally dry conditions are forecast, indicating very high water stress.",
        )

    if prediction <= -1.5:
        return (
            "Severe Water Stress",
            "Significantly drier-than-normal conditions are forecast.",
        )

    if prediction <= -1.0:
        return (
            "High Water Stress",
            "Dry conditions are forecast, with elevated water stress.",
        )

    if prediction <= -0.5:
        return (
            "Moderate Water Stress",
            "Slightly drier-than-normal conditions are forecast.",
        )

    if prediction < 0.5:
        return (
            "Near Normal",
            "Water conditions are forecast to remain close to the historical monthly norm.",
        )

    if prediction < 1.0:
        return (
            "Low Water Stress",
            "Wetter-than-normal conditions are forecast, suggesting relatively low water stress.",
        )

    if prediction <= 2.0:
        return (
            "Very Low Water Stress",
            "Substantially wetter-than-normal conditions are forecast.",
        )

    return (
        "Exceptionally Wet",
        "Exceptionally wet conditions are forecast relative to the historical climatology.",
    )


_cached_live_forecast: dict | None = None
_cached_observation_month = None


def generate_live_forecast() -> dict:
    """
    Fetch the latest observations, construct the
    trained model's features and generate a
    one-month-ahead WSI forecast.

    The result is cached for the latest completed observation
    month. Each request first checks the latest common completed
    month available from both CHIRPS and ERA5-Land. If that month
    has changed, a fresh forecast is generated automatically.
    """

    global _cached_live_forecast
    global _cached_observation_month

    latest_observation_month = latest_common_complete_month()

    if (
        _cached_live_forecast is not None
        and _cached_observation_month == latest_observation_month
    ):
        return _cached_live_forecast

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # -----------------------------------------
    # Load trained model
    # -----------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    # -----------------------------------------
    # Fetch current environmental observations
    # -----------------------------------------

    monthly = fetch_latest_monthly_data()

    # -----------------------------------------
    # Build 15-feature model input
    # -----------------------------------------

    model_input, observation_month = (
        build_latest_model_input(
            monthly
        )
    )

    # Explicitly preserve training feature order.
    model_input = model_input[
        MODEL_FEATURES
    ]

    # -----------------------------------------
    # Generate forecast
    # -----------------------------------------

    prediction = float(
        model.predict(
            model_input
        )[0]
    )

    forecast_month = (
        forecast_month_from_observation(
            observation_month
        )
    )

    category, description = (
        classify_water_stress(
            prediction
        )
    )

    latest = monthly.iloc[-1]

    # -----------------------------------------
    # API-friendly result
    # -----------------------------------------

    result = {
        "observation_month":
            observation_month.strftime(
                "%Y-%m"
            ),

        "forecast_month":
            forecast_month.strftime(
                "%Y-%m"
            ),

        "water_stress_index":
            prediction,

        "category":
            category,

        "description":
            description,

        "latest_indicators": {
            "precipitation_mm":
                float(
                    latest[
                        "precipitation_mm"
                    ]
                ),

            "pet_mm":
                float(
                    latest[
                        "pet_mm"
                    ]
                ),

            "temperature_max_c":
                float(
                    latest[
                        "temperature_max_c"
                    ]
                ),

            "soil_moisture_layer1":
                float(
                    latest[
                        "soil_moisture_layer1"
                    ]
                ),

            "soil_moisture_layer2":
                float(
                    latest[
                        "soil_moisture_layer2"
                    ]
                ),

            "runoff_mm":
                float(
                    latest[
                        "runoff_mm"
                    ]
                ),

            "solar_radiation":
                float(
                    latest[
                        "solar_radiation"
                    ]
                ),
        },

        "model_features": {
            feature:
                float(
                    model_input.iloc[0][
                        feature
                    ]
                )
            for feature in MODEL_FEATURES
        },
    }

    _cached_live_forecast = result
    _cached_observation_month = observation_month

    return result


def clear_live_forecast_cache() -> None:
    """
    Clear the cached live forecast so the next request
    retrieves fresh Earth Engine observations.
    """

    global _cached_live_forecast
    global _cached_observation_month

    _cached_live_forecast = None
    _cached_observation_month = None


if __name__ == "__main__":

    print(
        "\n========================================"
    )
    print(
        "LIVE WATER STRESS FORECAST"
    )
    print(
        "========================================"
    )

    initialize_earth_engine()

    result = (
        generate_live_forecast()
    )

    print(
        "\nObservation month:",
        result[
            "observation_month"
        ],
    )

    print(
        "Forecast month:",
        result[
            "forecast_month"
        ],
    )

    print(
        "\nPredicted Water Stress Index:",
        round(
            result[
                "water_stress_index"
            ],
            4,
        ),
    )

    print(
        "Forecast category:",
        result[
            "category"
        ],
    )

    print(
        "\nDescription:"
    )

    print(
        result[
            "description"
        ]
    )

    print(
        "\nLatest environmental indicators:"
    )

    for key, value in (
        result[
            "latest_indicators"
        ].items()
    ):
        print(
            f"  {key}: {value:.6f}"
        )