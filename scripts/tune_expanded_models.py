from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# EXPANDED SWBA HYPERPARAMETER TUNING
# Model selection: Training + Validation
# Test period: NOT used for tuning
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "processed" / "expanded"
    / "mnjoli_expanded_forecasting_dataset_1981_2025.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "expanded" / "hyperparameter_tuning"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

df["target_month"] = pd.to_datetime(df["target_month"])

TARGET = "target_swba_3month"

EXCLUDED = [
    "forecast_origin_month",
    "target_month",
    TARGET,
    "dataset_split",
]

FEATURES = [
    col for col in df.columns
    if col not in EXCLUDED
]

train = df[df["dataset_split"] == "train"].copy()
validation = df[df["dataset_split"] == "validation"].copy()

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

print("=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

print(f"Training observations: {len(train)}")
print(f"Validation observations: {len(validation)}")
print(f"Features: {len(FEATURES)}")

# ------------------------------------------------------------
# 2. Metric function
# ------------------------------------------------------------

def evaluate(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }

results = []

# ------------------------------------------------------------
# 3. RIDGE REGRESSION
# ------------------------------------------------------------

print("\nTuning Ridge Regression...")

ridge_alphas = [
    0.001, 0.01, 0.1, 1.0, 10.0,
    100.0, 500.0, 1000.0
]

for alpha in ridge_alphas:

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_val)

    metrics = evaluate(y_val, pred)

    results.append({
        "Model": "Ridge",
        "Parameters": f"alpha={alpha}",
        "Validation_MAE": metrics["MAE"],
        "Validation_RMSE": metrics["RMSE"],
        "Validation_R2": metrics["R2"],
    })

# ------------------------------------------------------------
# 4. ELASTIC NET
# ------------------------------------------------------------

print("Tuning Elastic Net...")

elastic_alphas = [
    0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0
]

elastic_l1_ratios = [
    0.1, 0.3, 0.5, 0.7, 0.9, 1.0
]

for alpha in elastic_alphas:

    for l1_ratio in elastic_l1_ratios:

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=30000,
                random_state=42
            )),
        ])

        model.fit(X_train, y_train)

        pred = model.predict(X_val)

        metrics = evaluate(y_val, pred)

        results.append({
            "Model": "ElasticNet",
            "Parameters": (
                f"alpha={alpha}, "
                f"l1_ratio={l1_ratio}"
            ),
            "Validation_MAE": metrics["MAE"],
            "Validation_RMSE": metrics["RMSE"],
            "Validation_R2": metrics["R2"],
        })

# ------------------------------------------------------------
# 5. XGBOOST
# ------------------------------------------------------------

print("Tuning XGBoost...")

try:

    from xgboost import XGBRegressor

    xgb_params = []

    for max_depth in [2, 3, 4]:

        for learning_rate in [0.01, 0.03, 0.05]:

            for n_estimators in [200, 400, 600]:

                for reg_lambda in [1, 5, 10]:

                    xgb_params.append({
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                        "reg_lambda": reg_lambda,
                    })

    total = len(xgb_params)

    for i, params in enumerate(xgb_params, start=1):

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                **params,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )),
        ])

        model.fit(X_train, y_train)

        pred = model.predict(X_val)

        metrics = evaluate(y_val, pred)

        results.append({
            "Model": "XGBoost",
            "Parameters": str(params),
            "Validation_MAE": metrics["MAE"],
            "Validation_RMSE": metrics["RMSE"],
            "Validation_R2": metrics["R2"],
        })

        if i % 25 == 0:

            print(
                f"XGBoost configurations evaluated: "
                f"{i}/{total}"
            )

except ImportError:

    print(
        "XGBoost unavailable. "
        "Install it with: pip install xgboost"
    )

# ------------------------------------------------------------
# 6. Rank all configurations by validation RMSE
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Validation_RMSE"
).reset_index(drop=True)

output_file = (
    OUTPUT_DIR
    / "hyperparameter_tuning_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# 7. Report best configuration for each model
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST VALIDATION CONFIGURATION BY MODEL")
print("=" * 70)

for model_name in results_df["Model"].unique():

    subset = results_df[
        results_df["Model"] == model_name
    ]

    best = subset.iloc[0]

    print(f"\n{model_name}")

    print(f"Parameters: {best['Parameters']}")

    print(
        f"Validation MAE:  "
        f"{best['Validation_MAE']:.4f}"
    )

    print(
        f"Validation RMSE: "
        f"{best['Validation_RMSE']:.4f}"
    )

    print(
        f"Validation R2:   "
        f"{best['Validation_R2']:.4f}"
    )

# ------------------------------------------------------------
# 8. Show overall top 10 configurations
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 CONFIGURATIONS OVERALL")
print("=" * 70)

print(
    results_df.head(10).round(4).to_string(
        index=False
    )
)

print("\nAll tuning results saved to:")
print(output_file)

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING COMPLETE")
print("=" * 70)
