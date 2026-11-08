from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded"
    / "mnjoli_expanded_forecasting_dataset_1981_2025.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "expanded"
    / "final_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_swba_3month"
PERSISTENCE_FEATURE = "swba_3month_lag_0"

TEST_START = "2020-01-01"
TEST_END = "2025-12-31"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL CONSOLIDATED EVALUATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

# Identify target dates
if "target_month" not in df.columns:
    raise ValueError(
        "The dataset does not contain 'target_month'. "
        "Please inspect the dataset columns."
    )

df["target_month"] = pd.to_datetime(df["target_month"])

if TARGET not in df.columns:
    raise ValueError(f"Missing target column: {TARGET}")

if PERSISTENCE_FEATURE not in df.columns:
    raise ValueError(
        f"Missing persistence feature: {PERSISTENCE_FEATURE}"
    )

# Remove rows without target values
df = df.dropna(subset=[TARGET]).copy()

# Development data: before January 2020
development = df[
    df["target_month"] < TEST_START
].copy()

# Final evaluation period: January 2020 through December 2025
test = df[
    (df["target_month"] >= TEST_START)
    & (df["target_month"] <= TEST_END)
].copy()

development = development.sort_values("target_month")
test = test.sort_values("target_month")

y_dev = development[TARGET].astype(float)
y_test = test[TARGET].astype(float)

print(f"\nDevelopment observations: {len(development)}")
print(
    "Development period:",
    development["target_month"].min().strftime("%Y-%m"),
    "to",
    development["target_month"].max().strftime("%Y-%m"),
)

print(f"\nTest observations: {len(test)}")
print(
    "Test period:",
    test["target_month"].min().strftime("%Y-%m"),
    "to",
    test["target_month"].max().strftime("%Y-%m"),
)

# ============================================================
# FEATURE SELECTION
# ============================================================

excluded_columns = {
    TARGET,
    "target_month",
    "forecast_origin_month",
    "month_date",
    "date",
    "year",
    "split",
    "partition",
    "dataset_split",
}

# Use numeric predictors only.
feature_columns = [
    col
    for col in development.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(development[col])
]

X_dev = development[feature_columns].copy()
X_test = test[feature_columns].copy()

print(f"\nNumber of predictors: {len(feature_columns)}")

# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {
    "Lasso": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Lasso(
            alpha=0.01,
            max_iter=50000
        )),
    ]),

    "Ridge": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(
            alpha=10.0
        )),
    ]),
}

# Tuned XGBoost configuration from development validation
try:
    from xgboost import XGBRegressor

    models["Tuned XGBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            max_depth=4,
            learning_rate=0.03,
            n_estimators=600,
            reg_lambda=5,
            objective="reg:squarederror",
            random_state=42,
        )),
    ])

except ImportError:
    print(
        "\nWARNING: XGBoost is not installed. "
        "The XGBoost model will be skipped."
    )

# ============================================================
# FIT MODELS AND GENERATE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "target_month": test["target_month"].values,
    "observed_swba": y_test.values,
})

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_dev, y_dev)

    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    print(f"{name} completed.")

# ============================================================
# BASELINE 1: PERSISTENCE
# Predict next month's SWBA using the current month's SWBA
# ============================================================

predictions["Persistence"] = (
    test[PERSISTENCE_FEATURE].astype(float).values
)

# ============================================================
# BASELINE 2: SEASONAL CLIMATOLOGY
# Calculate average SWBA for each target calendar month
# using development data only.
# ============================================================

development_months = development["target_month"].dt.month
test_months = test["target_month"].dt.month

seasonal_means = (
    development.assign(
        calendar_month=development_months
    )
    .groupby("calendar_month")[TARGET]
    .mean()
)

global_development_mean = y_dev.mean()

predictions["Seasonal Climatology"] = (
    test_months.map(seasonal_means)
    .fillna(global_development_mean)
    .values
)

# ============================================================
# BASELINE 3: GLOBAL TRAINING MEAN
# ============================================================

predictions["Training-Mean Baseline"] = (
    global_development_mean
)

# ============================================================
# CALCULATE TEST METRICS
# ============================================================

results = []

model_names = [
    name for name in models.keys()
]

model_names += [
    "Persistence",
    "Seasonal Climatology",
    "Training-Mean Baseline",
]

for name in model_names:

    y_pred = predictions[name].values

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Test_MAE": mae,
        "Test_RMSE": rmse,
        "Test_R2": r2,
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Test_RMSE"
).reset_index(drop=True)

# ============================================================
# SAVE RESULTS
# ============================================================

metrics_path = OUTPUT_DIR / "final_test_metrics.csv"

predictions_path = (
    OUTPUT_DIR / "final_test_predictions.csv"
)

results_df.to_csv(metrics_path, index=False)

predictions.to_csv(
    predictions_path,
    index=False
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS: JANUARY 2020 – DECEMBER 2025")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Test_MAE": "{:.4f}".format,
            "Test_RMSE": "{:.4f}".format,
            "Test_R2": "{:.4f}".format,
        }
    )
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"\nMetrics:\n{metrics_path}")
print(f"\nPredictions:\n{predictions_path}")

print("\nEvaluation completed successfully.")
