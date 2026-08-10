# Machine Learning-Based Forecasting of Hydroclimatic Water Stress in Eswatini

A machine learning framework for **one-month-ahead forecasting of hydroclimatic water stress in Eswatini** using remotely sensed and reanalysis-derived environmental variables.

The project investigates whether hydroclimatic conditions observed at month **t** can provide useful predictive information about water stress at month **t+1**, with particular emphasis on chronological validation, leakage prevention, regularization, and comparison against simple forecasting baselines.

## Research Overview

Eswatini is vulnerable to recurrent drought, rainfall variability, increasing temperatures, and associated pressures on agriculture and water resources. Reliable short-term forecasting could complement conventional drought monitoring by providing advance information for climate-resilient water management.

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

The final modelling experiment uses **Ridge Regression** to forecast the water-stress index one month ahead.

## Forecasting Objective

The forecasting problem is formulated as:

> Use hydroclimatic information available at month **t** to predict the water-stress index at month **t+1**.

This distinction is important because the project is designed as a genuine forecasting experiment rather than a same-month estimation exercise.

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

## Water-Stress Index

The target variable is a standardized hydroclimatic water-stress index derived from the three-month accumulated climatic water balance.

The three-month water balance is standardized relative to the corresponding calendar month's climatology.

To prevent information from the validation and test periods from influencing the definition of the target, the final index uses climatological statistics calculated from the **training period only**.

Conceptually:

```text
WSI = (WB3 - training-month climatological mean) /
      training-month climatological standard deviation
```

where `WB3` represents the three-month accumulated climatic water balance.

This training-only climatology was retained throughout validation and independent testing.

## Leakage Prevention

Special attention was given to preventing temporal and target leakage.

The final workflow:

1. orders observations chronologically;
2. constructs the water-stress climatology from the training period only;
3. shifts the target one month forward;
4. performs feature ranking using training data only;
5. selects feature count using validation performance only;
6. tunes the Ridge regularization parameter using validation data only; and
7. leaves the 2024–2025 test period untouched until final model evaluation.

Variables directly involved in constructing the target were excluded from the predictor set where their inclusion could create target leakage.

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

## Final Model

The final forecasting model is **Ridge Regression** implemented as a preprocessing and modelling pipeline consisting of:

```text
Median Imputation
      ↓
Standard Scaling
      ↓
Ridge Regression
```

The regularization parameter was selected through validation-period experiments.

```text
Ridge alpha = 0.05
Number of predictors = 15
Forecast horizon = 1 month
```

Ridge Regression was selected because the environmental predictors exhibit substantial correlation arising from related hydroclimatic processes, temporal lags, and accumulated variables.

Regularization allows correlated predictors to be retained while constraining coefficient magnitude.

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
| Final Ridge t+1 | **1.3906** |
| Persistence | 1.7568 |
| Climatology | 1.7589 |

The Ridge model reduced RMSE by approximately:

- **20.84% relative to persistence**
- **20.93% relative to climatology**

These results indicate that the environmental predictors contain useful one-month-ahead predictive information beyond simply assuming that current water-stress conditions will persist into the following month.

## Extreme Events

Forecast performance deteriorated substantially during unusually extreme water-stress conditions.

Two independent-test observations had an absolute WSI of at least 2.

Performance was:

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

However, the observation was **not removed from the official test evaluation**. The complete 24-month test results remain the primary reported results.

This highlights an important limitation of the model: it provides useful predictive skill under more typical conditions but has difficulty reproducing rare hydroclimatic extremes.

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

## Repository Structure

```text
Eswatini-Water-Stress-ML/
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
├── notebooks/
│   └── 01_data_acquisition.ipynb
│
├── .gitignore
└── README.md
```

The repository structure may be expanded as the analysis is reorganized into separate data-preparation, modelling, evaluation, and visualization notebooks.

## Reproducibility

The project was developed in Python using Jupyter notebooks.

Core packages include:

```text
pandas
numpy
scikit-learn
matplotlib
```

A reproducible environment specification will be added to the repository as the project is finalized.

## Limitations

Several limitations should be considered when interpreting the results.

First, the monthly dataset is relatively small, restricting the complexity of models that can be estimated reliably.

Second, strong correlations exist among environmental predictors because temperature variables, precipitation accumulations, PET, runoff, soil moisture, and their temporal lags represent interconnected physical processes.

Third, the final model performs substantially worse during rare extreme conditions, particularly the May 2024 event.

Fourth, the current modelling framework represents temporally aggregated hydroclimatic conditions and should not be interpreted as a complete representation of local water availability or operational water scarcity.

Finally, Ridge coefficients describe predictive associations within the fitted model and should not be interpreted as evidence of causal hydroclimatic relationships.

## Research Significance

The project demonstrates a leakage-aware framework for evaluating whether remotely sensed and reanalysis-derived environmental information can support short-term water-stress forecasting in Eswatini.

The results suggest that a relatively simple regularized linear model can outperform persistence and climatological benchmarks while remaining computationally efficient and comparatively interpretable.

The work contributes toward the broader development of data-driven early-warning approaches for climate-resilient water-resource management in data-constrained environments.

## Current Status

The current repository contains the final one-month-ahead Ridge forecasting experiment and associated datasets.

Current final model:

```text
Model:            Ridge Regression
Forecast horizon: 1 month
Predictors:       15
Alpha:            0.05

Independent test:
January 2024 – December 2025
N = 24

MAE:  0.9625
RMSE: 1.3906
R²:   0.2124

RMSE improvement over persistence: 20.84%
RMSE improvement over climatology: 20.93%
```

## Author

**Paradise Sengeto Nxumalo**

Bachelor of Computer Science, First Class Honours  
Eswatini

## Citation

This repository supports an ongoing research manuscript on machine-learning-based hydroclimatic water-stress forecasting in Eswatini.

A formal citation and DOI will be added following publication or archival release.

## License

A formal open-source and data license will be specified before the repository is released as the permanent research archive.