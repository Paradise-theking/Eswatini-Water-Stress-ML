from pathlib import Path

import joblib
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from backend.live_forecast import (
    generate_live_forecast,
    clear_live_forecast_cache,
)
from backend.data_ingestion import (
    initialize_earth_engine,
    fetch_era5_daily,
    fetch_chirps_daily,
    transform_daily_data,
    aggregate_monthly,
)
from backend.live_history import (
    fetch_live_observed_history,
    TRAINING_CLIMATOLOGY,
)


# ---------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------

app = FastAPI(
    title="Eswatini Water Stress Forecast API",
    description="Machine-learning API for forecasting water stress in Eswatini.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "water_stress_ridge.joblib"
DATA_PATH = BASE_DIR.parent / "data" / "processed" / "mnjoli_water_stress_ml_dataset.csv"

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(
        f"Could not load model from {MODEL_PATH}: {exc}"
    ) from exc


# Exact feature order used during model training
FEATURES = [
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


# ---------------------------------------------------------
# REQUEST SCHEMA
# ---------------------------------------------------------

class WaterStressInput(BaseModel):
    soil_moisture_layer1_lag1: float
    precipitation_mm_lag1: float = Field(ge=0)
    pet_mm: float = Field(ge=0)

    soil_moisture_layer2_lag1: float
    precipitation_mm: float = Field(ge=0)
    soil_moisture_layer2: float

    precipitation_3month: float = Field(ge=0)
    soil_moisture_layer1: float
    pet_mm_lag1: float = Field(ge=0)

    temperature_max_c: float

    surface_runoff_mm_lag1: float = Field(ge=0)
    runoff_mm_lag1: float = Field(ge=0)

    pet_3month: float = Field(ge=0)

    solar_radiation: float = Field(ge=0)
    solar_radiation_lag1: float = Field(ge=0)


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "success",
        "message": "Eswatini Water Stress Forecast API is running",
        "model": "Ridge Regression",
        "features": len(FEATURES),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }
@app.get("/history")
def get_history():
    try:
        df = pd.read_csv(DATA_PATH)

        history = (
            df[["month_date", "water_stress_index"]]
            .rename(
                columns={
                    "month_date": "date",
                    "water_stress_index": "wsi"
                }
            )
        )

        return {
            "status": "success",
            "count": len(history),
            "start_date": history["date"].iloc[0],
            "end_date": history["date"].iloc[-1],
            "data": history.to_dict(orient="records")
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load historical data: {str(exc)}"
        )

@app.get("/history/live")
def get_live_history():
    """
    Temporary diagnostic endpoint for validating live SWBA history.
    """
    try:
        print("LIVE HISTORY: starting", flush=True)

        print("LIVE HISTORY: initializing Earth Engine", flush=True)
        initialize_earth_engine()
        print("LIVE HISTORY: Earth Engine initialized", flush=True)

        print("LIVE HISTORY: fetching observed history", flush=True)
        history = fetch_live_observed_history()
        print(
            f"LIVE HISTORY: history fetched, rows={len(history)}",
            flush=True,
        )

        if history.empty:
            print("LIVE HISTORY: history is empty", flush=True)

            return {
                "status": "success",
                "count": 0,
                "start_date": None,
                "end_date": None,
                "data": [],
            }

        print("LIVE HISTORY: converting rows to response", flush=True)

        data = []

        for _, row in history.iterrows():
            data.append(
                {
                    "date": row[
                        "month_date"
                    ].strftime("%Y-%m-%d"),
                    "swba": float(
                        row["swba"]
                    ),
                    "water_balance_mm": float(
                        row["water_balance_mm"]
                    ),
                    "water_balance_3month": float(
                        row["water_balance_3month"]
                    ),
                }
            )

        print(
            f"LIVE HISTORY: response prepared, rows={len(data)}",
            flush=True,
        )

        return {
            "status": "success",
            "count": len(data),
            "start_date": data[0]["date"],
            "end_date": data[-1]["date"],
            "data": data,
        }

    except Exception as exc:
        print(
            f"LIVE HISTORY ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not generate live "
                f"SWBA history: {str(exc)}"
            ),
        )

@app.get("/history/live/test")
def test_live_history():
    try:
        print("LIVE TEST: starting", flush=True)

        initialize_earth_engine()
        print("LIVE TEST: Earth Engine initialized", flush=True)

        history = fetch_live_observed_history()
        print(
            f"LIVE TEST: fetched {len(history)} rows",
            flush=True,
        )

        if history.empty:
            return {
                "status": "success",
                "count": 0,
                "data": [],
            }

        data = [
            {
                "date": row["month_date"].strftime("%Y-%m-%d"),
                "swba": float(row["swba"]),
            }
            for _, row in history.iterrows()
        ]

        print(
            f"LIVE TEST: response prepared, rows={len(data)}",
            flush=True,
        )

        return {
            "status": "success",
            "count": len(data),
            "data": data,
        }

    except Exception as exc:
        print(
            f"LIVE TEST ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )    

@app.get("/history/live/validate-dec2025")
def validate_dec2025():
    """
    Temporary validation endpoint.

    Recalculates December 2025 SWBA using the live
    Earth Engine ingestion pipeline and compares it
    with the official research value.
    """
    try:
        print(
            "VALIDATION: starting December 2025 validation",
            flush=True,
        )

        initialize_earth_engine()

        print(
            "VALIDATION: Earth Engine initialized",
            flush=True,
        )

        start_date = "2025-10-01"
        end_date = "2025-12-31"

        print(
            f"VALIDATION: fetching {start_date} to {end_date}",
            flush=True,
        )

        era5 = fetch_era5_daily(
            start_date,
            end_date,
        )

        print(
            f"VALIDATION: ERA5 rows = {len(era5)}",
            flush=True,
        )

        chirps = fetch_chirps_daily(
            start_date,
            end_date,
        )

        print(
            f"VALIDATION: CHIRPS rows = {len(chirps)}",
            flush=True,
        )

        daily = transform_daily_data(
            era5,
            chirps,
        )

        print(
            f"VALIDATION: transformed rows = {len(daily)}",
            flush=True,
        )

        monthly = aggregate_monthly(
            daily,
        )

        print(
            f"VALIDATION: monthly rows = {len(monthly)}",
            flush=True,
        )

        monthly = (
            monthly
            .copy()
            .sort_values("month_date")
            .reset_index(drop=True)
        )

        monthly["water_balance_mm"] = (
            monthly["precipitation_mm"]
            - monthly["pet_mm"]
        )

        monthly["water_balance_3month"] = (
            monthly["water_balance_mm"]
            .rolling(
                window=3,
                min_periods=3,
            )
            .sum()
        )

        monthly["calendar_month"] = (
            pd.to_datetime(
                monthly["month_date"]
            ).dt.month
        )

        monthly["wb3_train_mean"] = (
            monthly["calendar_month"]
            .map(
                {
                    month: values["mean"]
                    for month, values
                    in TRAINING_CLIMATOLOGY.items()
                }
            )
        )

        monthly["wb3_train_std"] = (
            monthly["calendar_month"]
            .map(
                {
                    month: values["std"]
                    for month, values
                    in TRAINING_CLIMATOLOGY.items()
                }
            )
        )

        monthly["swba"] = (
            (
                monthly["water_balance_3month"]
                - monthly["wb3_train_mean"]
            )
            / monthly["wb3_train_std"]
        )

        result = monthly[
            pd.to_datetime(monthly["month_date"])
            == pd.Timestamp("2025-12-01")
        ]

        if result.empty:
            raise ValueError(
                "December 2025 was not found in "
                "the aggregated live data."
            )

        row = result.iloc[0]

        live_swba = float(row["swba"])
        official_swba = 1.470839
        difference = live_swba - official_swba

        print(
            f"VALIDATION: live SWBA = {live_swba}",
            flush=True,
        )

        print(
            f"VALIDATION: official SWBA = {official_swba}",
            flush=True,
        )

        print(
            f"VALIDATION: difference = {difference}",
            flush=True,
        )

        return {
            "status": "success",
            "month": "2025-12-01",
            "live_pipeline": {
                "precipitation_mm": float(
                    row["precipitation_mm"]
                ),
                "pet_mm": float(
                    row["pet_mm"]
                ),
                "water_balance_mm": float(
                    row["water_balance_mm"]
                ),
                "water_balance_3month": float(
                    row["water_balance_3month"]
                ),
                "training_mean": float(
                    row["wb3_train_mean"]
                ),
                "training_std": float(
                    row["wb3_train_std"]
                ),
                "swba": live_swba,
            },
            "official_research": {
                "swba": official_swba,
            },
            "comparison": {
                "difference": difference,
                "absolute_difference": abs(difference),
            },
        }

    except Exception as exc:
        print(
            f"VALIDATION ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

@app.get("/forecast/latest")
def forecast_latest():
    try:
        df = pd.read_csv(DATA_PATH)

        latest_row = df.iloc[-1]

        input_data = pd.DataFrame(
            [{
                feature: latest_row[feature]
                for feature in FEATURES
            }],
            columns=FEATURES
        )

        prediction = model.predict(input_data)
        predicted_value = float(prediction[0])

        latest_date = str(latest_row["month_date"])

        return {
            "status": "success",
            "input_date": latest_date,
            "prediction": predicted_value,
            "features": {
                feature: float(latest_row[feature])
                for feature in FEATURES
            }
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Latest forecast failed: {str(exc)}"
        )

@app.post("/forecast/live/refresh")
def refresh_live_forecast():
    """
    Clear the cached forecast and regenerate it
    using the latest available Earth Engine data.
    """
    try:
        clear_live_forecast_cache()

        initialize_earth_engine()

        result = generate_live_forecast()

        return {
            "status": "success",
            "refreshed": True,
            **result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Live forecast refresh failed: {str(exc)}",
        )
    
@app.get("/forecast/live")
def live_water_stress_forecast():
    """
    Generate a one-month-ahead forecast using the latest
    commonly available ERA5-Land and CHIRPS observations.
    """
    try:
        initialize_earth_engine()

        result = generate_live_forecast()

        return {
            "status": "success",
            **result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Live forecast failed: {str(exc)}",
        )
    
@app.post("/predict")
def predict_water_stress(data: WaterStressInput):
    try:
        input_data = pd.DataFrame(
            [data.model_dump()],
            columns=FEATURES
        )

        prediction = model.predict(input_data)

        predicted_value = float(prediction[0])

        return {
            "status": "success",
            "prediction": predicted_value
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )