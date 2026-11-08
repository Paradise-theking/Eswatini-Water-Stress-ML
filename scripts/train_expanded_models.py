from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ============================================================
# EXPANDED SWBA MODEL COMPARISON
# Period: 1981–2025
# Target: Next month's 3-month SWBA
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "expanded"
    / "mnjoli_expanded_forecasting_dataset_1981_2025.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "expanded"
    / "model_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

df["forecast_origin_month"] = pd.to_datetime(
    df["forecast_origin_month"]
)

df["target_month"] = pd.to_datetime(
    df["target_month"]
)

TARGET = "target_swba_3month"

SPLIT = "dataset_split"

EXCLUDED_COLUMNS = [
    "forecast_origin_month",
    "target_month",
    TARGET,
    SPLIT,
]

FEATURES = [
    col for col in df.columns
    if col not in EXCLUDED_COLUMNS
]

train = df[df[SPLIT] == "train"].copy()
validation = df[df[SPLIT] == "validation"].copy()
test = df[df[SPLIT] == "test"].copy()

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print("=" * 70)
print("EXPANDED SWBA MODEL COMPARISON")
print("=" * 70)

print(f"Training observations:   {len(train)}")
print(f"Validation observations:{len(validation)}")
print(f"Testing observations:   {len(test)}")
print(f"Number of features:     {len(FEATURES)}")

# ------------------------------------------------------------
# 2. Define conventional ML models
# ------------------------------------------------------------

models = {

    "Ridge": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),

    "ElasticNet": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=20000,
            random_state=42
        )),
    ]),

    "SVR": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVR(
            kernel="rbf",
            C=10,
            epsilon=0.1,
            gamma="scale"
        )),
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )),
    ]),

    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )),
    ]),
}

# ------------------------------------------------------------
# 3. Add XGBoost if installed
# ------------------------------------------------------------

try:

    from xgboost import XGBRegressor

    models["XGBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )),
    ])

except ImportError:

    print(
        "\nWARNING: XGBoost is not installed."
        "\nIt will be skipped."
        "\nInstall using: pip install xgboost"
    )

# ------------------------------------------------------------
# 4. Define evaluation metrics
# ------------------------------------------------------------

def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(
            actual, predicted
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                actual, predicted
            )
        ),

        "R2": r2_score(
            actual, predicted
        ),
    }


# ------------------------------------------------------------
# 5. Train using training set
#    Use validation set for preliminary selection
# ------------------------------------------------------------

results = []
predictions = {}

for name, pipeline in models.items():

    print(f"\nTraining {name}...")

    pipeline.fit(
        X_train,
        y_train
    )

    val_pred = pipeline.predict(X_val)

    test_pred = pipeline.predict(X_test)

    val_metrics = calculate_metrics(
        y_val,
        val_pred
    )

    test_metrics = calculate_metrics(
        y_test,
        test_pred
    )

    results.append({
        "Model": name,

        "Validation_MAE": val_metrics["MAE"],
        "Validation_RMSE": val_metrics["RMSE"],
        "Validation_R2": val_metrics["R2"],

        "Test_MAE": test_metrics["MAE"],
        "Test_RMSE": test_metrics["RMSE"],
        "Test_R2": test_metrics["R2"],
    })

    predictions[name] = test_pred

    print(
        f"Validation RMSE: "
        f"{val_metrics['RMSE']:.4f}"
    )

    print(
        f"Test RMSE: "
        f"{test_metrics['RMSE']:.4f}"
    )

# ------------------------------------------------------------
# 6. Persistence baseline
#
# Predict next month's SWBA using current month's SWBA.
# ------------------------------------------------------------

persistence_feature = "swba_3month_lag_0"

if persistence_feature not in FEATURES:

    raise ValueError(
        "Persistence feature not found: "
        + persistence_feature
    )

persistence_val = (
    X_val[persistence_feature].to_numpy()
)

persistence_test = (
    X_test[persistence_feature].to_numpy()
)

val_metrics = calculate_metrics(
    y_val,
    persistence_val
)

test_metrics = calculate_metrics(
    y_test,
    persistence_test
)

results.append({
    "Model": "Persistence",

    "Validation_MAE": val_metrics["MAE"],
    "Validation_RMSE": val_metrics["RMSE"],
    "Validation_R2": val_metrics["R2"],

    "Test_MAE": test_metrics["MAE"],
    "Test_RMSE": test_metrics["RMSE"],
    "Test_R2": test_metrics["R2"],
})

predictions["Persistence"] = persistence_test

# ------------------------------------------------------------
# 7. Climatology baseline
#
# Training-set mean SWBA is used as a simple baseline.
# No test observations are used to calculate the mean.
# ------------------------------------------------------------

climatology_value = y_train.mean()

climatology_val = np.full(
    len(y_val),
    climatology_value
)

climatology_test = np.full(
    len(y_test),
    climatology_value
)

val_metrics = calculate_metrics(
    y_val,
    climatology_val
)

test_metrics = calculate_metrics(
    y_test,
    climatology_test
)

results.append({
    "Model": "Climatology",

    "Validation_MAE": val_metrics["MAE"],
    "Validation_RMSE": val_metrics["RMSE"],
    "Validation_R2": val_metrics["R2"],

    "Test_MAE": test_metrics["MAE"],
    "Test_RMSE": test_metrics["RMSE"],
    "Test_R2": test_metrics["R2"],
})

predictions["Climatology"] = climatology_test

# ------------------------------------------------------------
# 8. Save results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Validation_RMSE"
).reset_index(drop=True)

results_file = (
    OUTPUT_DIR
    / "expanded_model_comparison_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

# ------------------------------------------------------------
# 9. Save test predictions
# ------------------------------------------------------------

prediction_df = pd.DataFrame({

    "target_month": test["target_month"].values,

    "actual_swba": y_test.values,

})

for name, values in predictions.items():

    prediction_df[
        f"{name}_prediction"
    ] = values

prediction_file = (
    OUTPUT_DIR
    / "expanded_model_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

# ------------------------------------------------------------
# 10. Display results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL COMPARISON RESULTS")
print("=" * 70)

print(
    results_df.round(4).to_string(
        index=False
    )
)

print("\nBest model based on validation RMSE:")

best_model = results_df.iloc[0]

print(
    f"{best_model['Model']} "
    f"(Validation RMSE = "
    f"{best_model['Validation_RMSE']:.4f})"
)

print("\nSaved results:")
print(results_file)

print("\nSaved test predictions:")
print(prediction_file)

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)
