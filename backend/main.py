from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from backend.live_forecast import (
    generate_live_forecast,
    clear_live_forecast_cache,
)
from backend.data_ingestion import initialize_earth_engine


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
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
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