Eswatini Water Stress ML

A deployed machine-learning system for one-month-ahead hydroclimatic water-stress forecasting in Eswatini using Earth-observation data.

Live Demo

Web application: https://eswatiniwaterstress.web.app

Production API: https://eswatini-water-stress-api.onrender.com

API health check: https://eswatini-water-stress-api.onrender.com/health

The Render free-tier backend may need a short cold-start period after inactivity.

Project Overview

This project investigates whether hydroclimatic conditions observed at month t can provide useful predictive information about hydroclimatic water stress at month t+1.

It combines two connected components:

Research pipeline — leakage-aware feature engineering, chronological model development, validation, independent testing, baseline comparison, residual analysis, and model interpretation.

Operational forecasting application — automated retrieval of newly available Earth-observation data, feature construction, model inference through FastAPI, and an interactive web dashboard.

The current production system uses a frozen Ridge Regression pipeline with 15 predictors and a one-month forecast horizon.

Data Sources

The system uses Earth-observation and reanalysis-derived environmental variables, primarily CHIRPS precipitation and ERA5-Land atmospheric/land-surface variables, plus monthly lags and three-month accumulated features. The operational backend retrieves the latest commonly available observations through Google Earth Engine and constructs the exact feature vector required by the trained model.

Water Stress Index

The target is a standardized hydroclimatic Water Stress Index (WSI) derived from the three-month accumulated climatic water balance.

Water Balance = Precipitation - Potential Evapotranspiration

Conceptually:

WSI = (WB3 - calendar-month climatological mean)
      ------------------------------------------
        calendar-month climatological standard deviation

Negative WSI values indicate drier-than-normal hydroclimatic conditions, values near zero indicate conditions near the climatological norm, and positive values indicate wetter-than-normal conditions. The continuous WSI is the primary scientific output; dashboard labels are an application-level interpretation layer.

Leakage-Aware Experimental Design

The final experiment orders observations chronologically, derives the target climatology from training data only, shifts the target one month forward, ranks predictors using training data only, selects feature count and Ridge regularization using validation data only, and leaves the 2024–2025 test period untouched until final evaluation.

Partition

Period

Training

July 2015 – December 2021

Validation

January 2022 – December 2023

Independent test

January 2024 – December 2025

The independent test contains 24 monthly forecasts.

Final Model

The final scikit-learn pipeline is:

Median Imputation -> Standard Scaling -> Ridge Regression

Parameter

Value

Model

Ridge Regression

Forecast horizon

1 month

Predictors

15

Ridge alpha

0.05

Selected Predictors

soil_moisture_layer1_lag1

precipitation_mm_lag1

pet_mm

soil_moisture_layer2_lag1

precipitation_mm

soil_moisture_layer2

precipitation_3month

soil_moisture_layer1

pet_mm_lag1

temperature_max_c

surface_runoff_mm_lag1

runoff_mm_lag1

pet_3month

solar_radiation

solar_radiation_lag1

Independent Test Results

Model

MAE

RMSE

R²

Final Ridge t+1

0.9625

1.3906

0.2124

Persistence

1.0290

1.7568

-0.2570

Climatology (WSI = 0)

1.1358

1.7589

-0.2600

The Ridge model reduced RMSE by 20.84% relative to persistence and 20.93% relative to climatology.

Extreme-Event Performance

Observation type

MAE

RMSE

Extreme (`

WSI

>= 2`)

2.9851

3.6728

Non-extreme

0.7786

0.9399

The largest error occurred in May 2024:

Observed WSI:   7.0614
Ridge forecast: 1.9365
Absolute error: 5.1248

The complete 24-month test period remains the official evaluation.

Production Forecasting System

Architecture

Google Earth Engine
CHIRPS + ERA5-Land
        |
        v
backend/data_ingestion.py
        |
        v
backend/feature_engineering.py
        |
        v
15-feature model input
        |
        v
models/water_stress_ridge.joblib
        |
        v
backend/live_forecast.py
        |
        v
FastAPI on Render
        |
        v
TypeScript/Vite frontend
        |
        v
Firebase Hosting

The frontend does not perform ML inference. Data acquisition, feature engineering, preprocessing, model loading, and prediction are handled by the backend.

Live Forecasting

The production backend initializes Google Earth Engine, checks the latest available observations, retrieves recent CHIRPS and ERA5-Land data, constructs lagged and accumulated predictors, preserves the trained model feature order, runs the frozen Ridge pipeline, and returns the next-month forecast and environmental indicators.

Research Dataset vs Live Forecast Data

Historical research dataset: July 2015 – December 2025, used for model development, independent evaluation, and historical visualization.

Live operational input: newer Earth-observation data retrieved through Google Earth Engine for current forecasting.

This separation preserves the frozen research experiment while allowing the deployed application to generate forecasts beyond the historical dataset.

API Endpoints

Method

Endpoint

Purpose

GET

/

API information

GET

/health

Backend/model health check

GET

/history

Historical monthly WSI observations

GET

/forecast/latest

Forecast from latest historical modelling row

GET

/forecast/live

Current Earth-Engine-based forecast

POST

/forecast/live/refresh

Clear cache and regenerate live forecast

POST

/predict

Predict from a supplied 15-feature observation

Deployment

Component

Platform

Frontend

Firebase Hosting

Backend API

Render

Environmental data

Google Earth Engine

ML model

scikit-learn Ridge pipeline

Repository Structure

Eswatini-Water-Stress-ML/
|-- backend/
|   |-- main.py
|   |-- data_ingestion.py
|   |-- feature_engineering.py
|   `-- live_forecast.py
|-- data/
|-- documentation/
|-- figures/
|-- frontend/
|-- models/
|   `-- water_stress_ridge.joblib
|-- notebooks/
|   |-- 01_data_acquisition.ipynb
|   |-- 02_data_preparation.ipynb
|   |-- 03_target_feature_engineering.ipynb
|   |-- 04_model_development.ipynb
|   `-- 05_final_evaluation.ipynb
|-- results/
|-- screenshots/
|-- requirements.txt
|-- firebase.json
`-- README.md

Running Locally

1. Clone

git clone https://github.com/Paradise-theking/Eswatini-Water-Stress-ML.git
cd Eswatini-Water-Stress-ML

2. Python Environment

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

3. Earth Engine Credentials

Configure your own Earth Engine-enabled Google Cloud service account:

$env:EE_SERVICE_ACCOUNT="your-service-account@your-project.iam.gserviceaccount.com"
$env:EE_PROJECT_ID="your-google-cloud-project-id"
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your-service-account-key.json"

Never commit service-account JSON keys or other credentials.

4. Start Backend

From the project root:

uvicorn backend.main:app --reload

5. Start Frontend

In a second terminal:

cd frontend
npm install
npm run dev

The frontend normally runs at http://localhost:5173.

Research Reproducibility

01_data_acquisition.ipynb
          |
          v
02_data_preparation.ipynb
          |
          v
03_target_feature_engineering.ipynb
          |
          v
04_model_development.ipynb
          |
          v
05_final_evaluation.ipynb

The frozen model specification is stored in data/processed/final_model_specification.json. Publication-oriented figures are available in figures/.

Key Technologies

Data/ML: Python, pandas, NumPy, scikit-learn, joblib, matplotlib, Google Earth Engine, CHIRPS, ERA5-Land
Backend: FastAPI, Uvicorn
Frontend/Deployment: TypeScript, Vite, Firebase Hosting, Render

Limitations

The monthly research dataset is relatively small.

Hydroclimatic predictors are strongly correlated.

Performance is substantially weaker during rare extreme events.

The WSI represents hydroclimatic conditions, not complete operational water scarcity.

Reservoir storage, groundwater, demand, abstraction, infrastructure, and socioeconomic exposure are not directly represented.

Ridge coefficients indicate predictive associations, not causality.

Dashboard categories are descriptive and are not official drought-warning thresholds.

Live forecasts depend on upstream Earth-observation availability and latency.

Free-tier cloud hosting can introduce backend cold-start delays.

Future Development

forecast archiving and prospective verification;

uncertainty-aware forecasting;

improved modelling of extreme events;

reservoir and hydrological data integration;

spatial forecasting across additional Eswatini catchments/regions;

comparison with additional time-series and ML approaches; and

operational warning-threshold calibration.

Version

The first stable deployed release is tagged v1.0.0.

Research Significance

The project demonstrates a complete research-to-deployment workflow for one-month-ahead hydroclimatic water-stress forecasting in Eswatini: Earth-observation data acquisition, leakage-aware feature engineering, independent evaluation, machine-learning inference, API deployment, and a public forecasting interface.

Author

Paradise Sengeto Nxumalo
Bachelor of Computer Science, First Class Honours
Eswatini

Citation

This repository supports ongoing work on machine-learning-based hydroclimatic water-stress forecasting in Eswatini. A formal citation and DOI can be added following publication or archival release.

License

A formal software and data license should be specified before the repository is used as a permanent public research archive.
