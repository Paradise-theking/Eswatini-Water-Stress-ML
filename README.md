# Machine Learning-Based Forecasting of Hydroclimatic Water Stress in Eswatini

A leakage-aware machine learning framework and interactive forecasting application for **one-month-ahead prediction of hydroclimatic water stress in Eswatini** using remotely sensed and reanalysis-derived environmental variables.

The project investigates whether hydroclimatic conditions observed at month **t** can provide useful predictive information about water stress at month **t+1**, with particular emphasis on chronological validation, leakage prevention, regularization, independent testing, and comparison against simple forecasting baselines.

The trained forecasting pipeline is also integrated into a **FastAPI backend** and **TypeScript/Vite dashboard** for interactive model inference and historical Water Stress Index visualization.

---

## Research Overview

Eswatini is vulnerable to recurrent drought, rainfall variability, increasing temperatures, and associated pressures on agriculture and water resources.

Reliable short-term hydroclimatic forecasting could complement conventional drought monitoring by providing advance information to support climate-resilient water-resource management.

This project develops a monthly water-stress forecasting framework using environmental variables including:

- precipitation;
- potential evapotranspiration (PET);
- temperature;
- soil moisture;
- runoff;
- surface runoff;
- solar radiation;
- dew point;
- wind speed; and
- lagged and accumulated hydroclimatic variables.

The final modelling experiment uses **Ridge Regression** to forecast the Water Stress Index one month ahead.

---

## Forecasting Objective

The forecasting problem is formulated as:

> Use hydroclimatic information available at month **t** to predict the Water Stress Index at month **t+1**.

This distinction is important because the project is designed as a genuine forecasting experiment rather than a same-month estimation exercise.

The forecasting horizon is therefore:

```text
Current hydroclimatic conditions (month t)
                    ↓
             Machine learning
                    ↓
      Water Stress Index (month t+1)
```

---

## Data

The analysis uses monthly environmental information derived from remotely sensed and reanalysis products covering Eswatini.

Primary data sources include:

- **CHIRPS** — precipitation;
- **ERA5-Land** — atmospheric and land-surface hydroclimatic variables; and
- derived temporal variables including monthly lags and rolling accumulations.

The modelling dataset contains variables representing precipitation, PET, temperature, dew point, soil moisture, runoff, surface runoff, solar radiation, wind speed, and related lagged predictors.

Raw/external and processed datasets used by the project are stored under:

```text
data/
├── external/
└── processed/
```

---

## Water Stress Index

The target variable is a standardized hydroclimatic Water Stress Index derived from the three-month accumulated climatic water balance.

Monthly climatic water balance is defined as:

```text
Water Balance = Precipitation - Potential Evapotranspiration
```

The three-month accumulated water balance is then standardized relative to the corresponding calendar month's climatology.

Conceptually:

```text
WSI = (WB3 - training-month climatological mean)
      ------------------------------------------------
        training-month climatological standard deviation
```

where `WB3` represents the three-month accumulated climatic water balance.

To prevent validation and test information from influencing the definition of the target, the final index uses climatological statistics calculated from the **training period only**.

The same training-period climatology is retained throughout validation and independent testing.

### Interpretation

Because the index represents standardized climatic water balance:

```text
Negative WSI  → drier-than-normal hydroclimatic conditions
WSI near 0    → conditions near the climatological norm
Positive WSI  → wetter-than-normal hydroclimatic conditions
```

A positive value should therefore **not** be interpreted as a percentage probability of water stress.

---

## Leakage Prevention

Special attention was given to preventing temporal and target leakage.

The final workflow:

1. orders observations chronologically;
2. constructs the Water Stress Index climatology from the training period only;
3. shifts the target one month forward;
4. performs feature ranking using training data only;
5. selects feature count using validation performance only;
6. tunes the Ridge regularization parameter using validation data only; and
7. leaves the 2024–2025 test period untouched until final model evaluation.

Variables directly involved in constructing the target were excluded from the predictor set where their inclusion could create target leakage.

---

## Chronological Evaluation

Random train/test splitting was avoided because neighbouring observations in hydroclimatic time series are temporally dependent.

The project instead uses chronological partitions.

The final independent evaluation covers:

```text
January 2024 – December 2025
```

with:

```text
N = 24 monthly forecasts
```

Training and validation observations preceding the test period were used to fit the final model after model configuration had been selected.

This design provides a more realistic assessment of future forecasting performance than randomly mixing earlier and later observations.

---

## Feature Selection

Candidate predictors were ranked according to their absolute relationship with the one-month-ahead target using **training data only**.

Different feature-set sizes were evaluated on the validation period.

The best validation performance was obtained using **15 predictors**.

### Final Selected Features

1. `soil_moisture_layer1_lag1`
2. `precipitation_mm_lag1`
3. `pet_mm`
4. `soil_moisture_layer2_lag1`
5. `precipitation_mm`
6. `soil_moisture_layer2`
7. `precipitation_3month`
8. `soil_moisture_layer1`
9. `pet_mm_lag1`
10. `temperature_max_c`
11. `surface_runoff_mm_lag1`
12. `runoff_mm_lag1`
13. `pet_3month`
14. `solar_radiation`
15. `solar_radiation_lag1`

---

## Final Model

The final forecasting model is **Ridge Regression** implemented as a preprocessing and modelling pipeline:

```text
Median Imputation
        ↓
Standard Scaling
        ↓
Ridge Regression
```

Final configuration:

```text
Model:            Ridge Regression
Ridge alpha:      0.05
Predictors:       15
Forecast horizon: 1 month
```

Ridge Regression was selected because the environmental predictors exhibit substantial correlation arising from related hydroclimatic processes, temporal lags, and accumulated variables.

Regularization allows correlated predictors to be retained while constraining coefficient magnitude.

The complete preprocessing and Ridge model are stored as a reusable scikit-learn pipeline for application inference.

---

## Independent Test Performance

The final Ridge model was evaluated exclusively on the independent **January 2024–December 2025** test period.

| Metric | Final Ridge |
|---|---:|
| MAE | 0.9625 |
| RMSE | 1.3906 |
| R² | 0.2124 |

The model was also compared against two simple forecasting benchmarks.

| Model | RMSE |
|---|---:|
| **Final Ridge t+1** | **1.3906** |
| Persistence | 1.7568 |
| Climatology | 1.7589 |

The Ridge model reduced RMSE by approximately:

- **20.84% relative to persistence**
- **20.93% relative to climatology**

These results indicate that the environmental predictors contain useful one-month-ahead predictive information beyond simply assuming that current water-stress conditions will persist into the following month.

---

## Extreme Events

Forecast performance deteriorated substantially during unusually extreme Water Stress Index conditions.

Two independent-test observations had an absolute WSI of at least 2.

| Observation Type | MAE | RMSE |
|---|---:|---:|
| Extreme observations | 2.9851 | 3.6728 |
| Non-extreme observations | 0.7786 | 0.9399 |

The largest error occurred in **May 2024**, when:

```text
Observed WSI = 7.0614
Ridge forecast = 1.9365
Absolute error = 5.1248
```

A sensitivity analysis excluding this observation produced:

```text
MAE  = 0.7815
RMSE = 0.9360
```

However, the observation was **not removed from the official test evaluation**.

The complete 24-month test results remain the primary reported results.

This highlights an important limitation of the model: it provides useful predictive skill under more typical conditions but has difficulty reproducing rare hydroclimatic extremes.

---

## Model Interpretation

Because all predictors are standardized before Ridge estimation, coefficient magnitudes provide a useful indication of relative model influence, although they should not be interpreted as causal effects.

The five largest absolute coefficients in the final model were:

| Rank | Feature | Coefficient |
|---:|---|---:|
| 1 | `soil_moisture_layer1_lag1` | 1.1918 |
| 2 | `pet_3month` | -0.9542 |
| 3 | `precipitation_mm` | 0.8025 |
| 4 | `pet_mm_lag1` | 0.7738 |
| 5 | `solar_radiation_lag1` | -0.6897 |

The model therefore draws predictive information from multiple components of the hydroclimatic system rather than from precipitation alone.

Coefficient signs should be interpreted cautiously because correlated predictors can redistribute explanatory weight across related variables.

---

# Interactive Forecasting Application

The trained Ridge pipeline is integrated into an interactive forecasting application consisting of:

```text
FastAPI backend
      +
TypeScript / Vite frontend
```

The purpose of the application is to translate the research model into a usable inference system while preserving the same trained preprocessing and Ridge pipeline.

---

## Application Architecture

```text
Processed hydroclimatic dataset
             │
             ▼
Latest available environmental observation
             │
             ▼
15 selected environmental features
             │
             ▼
Saved scikit-learn Pipeline
             │
             ├── Median Imputation
             ├── Standard Scaling
             └── Ridge Regression
             │
             ▼
        FastAPI Backend
             │
       ┌─────┴─────┐
       │           │
       ▼           ▼
 Historical WSI   Next-month forecast
       │           │
       └─────┬─────┘
             ▼
     TypeScript / Vite
          Dashboard
```

The frontend does **not** perform machine-learning inference.

Feature selection, model loading, preprocessing, and prediction are handled by the FastAPI backend.

---

## FastAPI Backend

The backend is implemented in:

```text
backend/main.py
```

It loads the trained scikit-learn pipeline and exposes model inference and historical information through REST endpoints.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API status message |
| `GET` | `/health` | Confirms that the trained ML pipeline is available |
| `GET` | `/history` | Returns the historical monthly WSI series |
| `GET` | `/forecast/latest` | Generates a forecast from the latest available feature observation |
| `POST` | `/predict` | Generates a prediction from a supplied 15-feature observation |

### Historical API

The `/history` endpoint currently exposes:

```text
Historical observations: 126 months
Start:                   July 2015
End:                     December 2025
Missing WSI values:      0
```

The endpoint supplies the historical series used by the frontend visualization.

### Latest Forecast API

`/forecast/latest` retrieves the most recent row in the processed modelling dataset, extracts the same 15 features required by the trained Ridge pipeline, and generates the corresponding one-month-ahead forecast.

This means the frontend does not contain a hard-coded ML feature vector.

---

## Interactive Dashboard

The frontend is implemented using **TypeScript and Vite**.

The dashboard presents:

- the forecast month;
- predicted Water Stress Index;
- interpreted hydroclimatic condition;
- environmental indicators used for inference;
- historical WSI observations;
- latest observed WSI;
- one-month-ahead forecast point;
- historical WSI visualization; and
- model metadata.

### Environmental Indicators

The dashboard dynamically displays environmental values returned by the forecasting API, including:

- monthly precipitation;
- three-month accumulated precipitation;
- top-layer soil moisture;
- deeper-layer soil moisture;
- maximum temperature; and
- potential evapotranspiration.

These values correspond to the feature observation used by the backend to generate the forecast.

---

## Historical Water Stress Visualization

The dashboard retrieves the historical Water Stress Index through `/history`.

The visualization displays:

```text
Historical WSI observations
        +
Climatological normal (WSI = 0)
        +
Next-month model forecast
```

The forecast is visually distinguished from the observed historical record so that a predicted value is not confused with an observation.

The current historical dataset contains **126 monthly observations from July 2015 through December 2025**.

---

## Dashboard Forecast Categories

For communication in the interactive dashboard, the continuous WSI output is mapped to descriptive hydroclimatic categories.

The current interpretation layer uses categories such as:

| WSI | Dashboard Interpretation |
|---:|---|
| ≤ -2.0 | Extreme Water Stress |
| -2.0 to -1.5 | Severe Water Stress |
| -1.5 to -1.0 | High Water Stress |
| -1.0 to -0.5 | Moderate Water Stress |
| -0.5 to +0.5 | Near Normal |
| +0.5 to +1.0 | Low Water Stress |
| +1.0 to +2.0 | Very Low Water Stress |
| > +2.0 | Exceptionally Wet |

**Important:** these categories form an application-level interpretation layer over the continuous WSI.

They should **not** currently be interpreted as independently validated operational drought-warning thresholds.

The continuous model prediction remains the primary scientific output.

---

## Forecast Date and Data Availability

The application determines the forecast month from the latest observation available in the processed dataset.

The current processed dataset ends in:

```text
December 2025
```

Because the model forecasts one month ahead, the latest available feature observation therefore produces a forecast for:

```text
January 2026
```

This distinction is important.

The current application demonstrates **model inference using the latest observation available in the research dataset**.

It does **not yet automatically retrieve live hydroclimatic observations for the present calendar month**.

Therefore, the January 2026 output should not be described as a live present-day forecast when the application is run at a later date.

A future operational implementation could extend the ingestion pipeline to retrieve newly available CHIRPS and ERA5-Land observations automatically before constructing the required model features.

---

## Repository Structure

```text
Eswatini-Water-Stress-ML/
│
├── backend/
│   └── main.py
│
├── data/
│   ├── external/
│   │   ├── mnjoli_chirps_daily_2015_2025.csv
│   │   ├── mnjoli_era5_land_daily_2015_2025_processed.csv
│   │   ├── mnjoli_era5_land_daily_2015_2025_raw.csv
│   │   └── mnjoli_w60e_catchment.geojson
│   │
│   └── processed/
│       ├── mnjoli_water_stress_features_daily.csv
│       ├── mnjoli_water_stress_ml_dataset.csv
│       └── mnjoli_water_stress_monthly.csv
│
├── documentation/
│   ├── research manuscript (.docx)
│   └── research manuscript (.pdf)
│
├── figures/
│   ├── independent_test_forecasts.png
│   ├── observed_vs_ridge_scatter.png
│   ├── ridge_standardized_coefficients.png
│   └── validation_rmse_feature_count.png
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── main.ts
│   │   └── style.css
│   ├── package.json
│   └── tsconfig.json
│
├── models/
│   └── final_ridge_pipeline.joblib
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_target_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_final_evaluation.ipynb
│
├── .gitignore
└── README.md
```

---

# Running the Application Locally

## Prerequisites

You will need:

- Python 3;
- pip;
- Node.js;
- npm; and
- Git if cloning the repository.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Paradise-theking/Eswatini-Water-Stress-ML.git
cd Eswatini-Water-Stress-ML
```

---

## 2. Backend Setup

Create a Python virtual environment from the project root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
pip install fastapi uvicorn pandas scikit-learn joblib
```

Move into the backend directory:

```powershell
cd backend
```

Start FastAPI:

```powershell
uvicorn main:app --reload
```

The API should become available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## 3. Frontend Setup

Open a second terminal and move into the frontend directory:

```powershell
cd frontend
```

Install the Node dependencies:

```powershell
npm install
```

Start the Vite development server:

```powershell
npm run dev
```

Vite will normally expose the application at:

```text
http://localhost:5173
```

Keep both the **FastAPI backend** and **Vite frontend** running while using the application.

---

# Research Reproducibility

The research workflow is organized across the Jupyter notebooks in approximately the following order:

```text
01_data_acquisition.ipynb
            ↓
02_data_preparation.ipynb
            ↓
03_target_feature_engineering.ipynb
            ↓
04_model_development.ipynb
            ↓
05_final_evaluation.ipynb
```

Core Python packages used across the research and application include:

```text
pandas
numpy
scikit-learn
matplotlib
joblib
FastAPI
Uvicorn
```

The frontend uses:

```text
TypeScript
Vite
```

For strict environment reproducibility, a dedicated Python dependency specification should be maintained alongside the repository.

---

## Research Figures

The repository contains final evaluation and interpretation figures under `figures/`, including:

- `independent_test_forecasts.png`
- `observed_vs_ridge_scatter.png`
- `ridge_standardized_coefficients.png`
- `validation_rmse_feature_count.png`

These figures document model performance, feature-set selection, observed-versus-predicted behaviour, and standardized Ridge coefficients.

---

## Limitations

Several limitations should be considered when interpreting the results.

### 1. Dataset Size

The monthly dataset is relatively small, restricting the complexity of models that can be estimated reliably.

### 2. Correlated Predictors

Strong correlations exist among environmental predictors because temperature variables, precipitation accumulations, PET, runoff, soil moisture, and their temporal lags represent interconnected physical processes.

### 3. Extreme Conditions

The final model performs substantially worse during rare extreme conditions, particularly the May 2024 event.

The model should therefore not be assumed to reproduce unprecedented or highly unusual hydroclimatic extremes reliably.

### 4. Hydroclimatic Rather Than Operational Water Scarcity

The modelling framework represents temporally aggregated hydroclimatic conditions.

It should not be interpreted as a complete representation of:

- reservoir storage;
- groundwater availability;
- water demand;
- distribution infrastructure;
- abstraction;
- agricultural demand; or
- operational water scarcity.

### 5. Predictive Rather Than Causal Interpretation

Ridge coefficients describe predictive associations within the fitted model.

They should not be interpreted as evidence of causal hydroclimatic relationships.

### 6. Current Data Availability

The interactive application currently performs inference using the latest observation already present in the research dataset.

It does not yet automatically download and preprocess newly released environmental observations.

### 7. Dashboard Categories

The descriptive dashboard categories are an interpretation layer and have not yet been independently calibrated as operational drought-warning thresholds for Eswatini.

---

## Future Development

Potential extensions include:

- automated ingestion of newly available CHIRPS precipitation data;
- automated ingestion of ERA5-Land variables;
- scheduled feature engineering and forecast generation;
- spatial forecasting beyond the current study representation;
- integration of reservoir and hydrological observations;
- probabilistic or uncertainty-aware forecasting;
- improved modelling of rare extreme events;
- comparison with additional time-series and machine-learning models;
- deployment of the FastAPI service;
- deployment of the interactive dashboard;
- forecast archiving and verification as new observations become available; and
- calibration of operational warning categories with domain and impact data.

---

## Research Significance

The project demonstrates a leakage-aware framework for evaluating whether remotely sensed and reanalysis-derived environmental information can support short-term water-stress forecasting in Eswatini.

The results suggest that a relatively simple regularized linear model can outperform persistence and climatological benchmarks while remaining computationally efficient and comparatively interpretable.

The work also demonstrates how a research forecasting pipeline can be translated into an interactive software application without moving preprocessing or machine-learning inference into the client.

The project contributes toward the broader development of data-driven early-warning approaches for climate-resilient water-resource management in data-constrained environments.

---

# Current Status

The repository currently includes both the completed one-month-ahead forecasting experiment and an interactive inference application.

## Research Model

```text
Model:             Ridge Regression
Forecast horizon:  1 month
Predictors:        15
Alpha:             0.05

Independent test:
January 2024 – December 2025
N = 24

MAE:   0.9625
RMSE:  1.3906
R²:    0.2124

RMSE improvement over persistence: 20.84%
RMSE improvement over climatology: 20.93%
```

## Application

```text
Backend:             FastAPI
Frontend:            TypeScript + Vite
Historical records:  126 monthly observations
Historical period:   July 2015 – December 2025
Forecast horizon:    1 month ahead
Latest input month:  December 2025
Current forecast:    January 2026
```

The application currently performs inference from the latest observation available in the research dataset.

Automated ingestion of newly available hydroclimatic observations remains future work.

---

## Author

**Paradise Sengeto Nxumalo**

Bachelor of Computer Science, First Class Honours  
Eswatini

---

## Citation

This repository supports an ongoing research manuscript on machine-learning-based hydroclimatic water-stress forecasting in Eswatini.

A formal citation and DOI will be added following publication or archival release.

---

## License

A formal open-source and data license will be specified before the repository is released as the permanent research archive.