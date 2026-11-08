
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed" / "expanded"
    / "mnjoli_expanded_forecasting_dataset_1981_2025.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "expanded"
    / "historical_2015_drought_event"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_swba_3month"
PERSISTENCE_FEATURE = "swba_3month_lag_0"

TRAIN_END = "2011-12-31"
VALIDATION_START = "2012-01-01"
EVENT_START = "2015-11-01"
EVENT_END = "2016-02-29"

DROUGHT_THRESHOLDS = [-1.0, -1.5, -2.0]

RANDOM_STATE = 42

# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

print("=" * 75)
print("HISTORICAL 2015–2016 DROUGHT EVENT EVALUATION")
print("=" * 75)

df = pd.read_csv(DATA_PATH)

df["target_month"] = pd.to_datetime(df["target_month"])
df["forecast_origin_month"] = pd.to_datetime(
    df["forecast_origin_month"]
)

required_columns = [
    TARGET,
    "target_month",
    "forecast_origin_month",
    PERSISTENCE_FEATURE,
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Remove rows without target values.
df = df.dropna(subset=[TARGET]).copy()

# ------------------------------------------------------------
# Historical training and validation
# ------------------------------------------------------------

training = df[
    df["target_month"] <= TRAIN_END
].copy()

validation = df[
    (df["target_month"] >= VALIDATION_START)
    & (df["target_month"] < EVENT_START)
].copy()

# All observations available before the drought event.
pre_event = df[
    df["target_month"] < EVENT_START
].copy()

# Four-month historical event.
event = df[
    (df["target_month"] >= EVENT_START)
    & (df["target_month"] <= EVENT_END)
].copy()

training = training.sort_values("target_month")
validation = validation.sort_values("target_month")
pre_event = pre_event.sort_values("target_month")
event = event.sort_values("target_month")

if len(training) == 0 or len(validation) == 0:
    raise ValueError("Training or validation dataset is empty.")

if len(event) != 4:
    raise ValueError(
        f"Expected 4 event observations, found {len(event)}."
    )

# ============================================================
# 3. FEATURE SELECTION
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

feature_columns = [
    col for col in training.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(training[col])
]

# Use the same feature columns throughout.
X_train = training[feature_columns].copy()
y_train = training[TARGET].astype(float)

X_validation = validation[feature_columns].copy()
y_validation = validation[TARGET].astype(float)

X_pre_event = pre_event[feature_columns].copy()
y_pre_event = pre_event[TARGET].astype(float)

X_event = event[feature_columns].copy()
y_event = event[TARGET].astype(float)

print("\nDATASET SUMMARY")
print("-" * 75)

print("Training observations:", len(training))
print("Validation observations:", len(validation))
print("Pre-event observations:", len(pre_event))
print("Event observations:", len(event))
print("Number of predictors:", len(feature_columns))

print(
    "\nTraining period:",
    training["target_month"].min().strftime("%Y-%m"),
    "to",
    training["target_month"].max().strftime("%Y-%m"),
)

print(
    "Validation period:",
    validation["target_month"].min().strftime("%Y-%m"),
    "to",
    validation["target_month"].max().strftime("%Y-%m"),
)

print(
    "Historical event:",
    event["target_month"].min().strftime("%Y-%m"),
    "to",
    event["target_month"].max().strftime("%Y-%m"),
)

if X_pre_event.isna().any().any():
    raise ValueError(
        "Missing values detected in pre-event predictors."
    )

if X_event.isna().any().any():
    raise ValueError(
        "Missing values detected in event predictors."
    )

# ============================================================
# 4. METRIC FUNCTION
# ============================================================

def evaluate_regression(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
    }


# ============================================================
# 5. HYPERPARAMETER TUNING
# Training: 1982–2011
# Validation: 2012–October 2015
# ============================================================

print("\n" + "=" * 75)
print("HYPERPARAMETER TUNING")
print("=" * 75)

tuning_results = []

# ------------------------------------------------------------
# RIDGE
# ------------------------------------------------------------

print("\nTuning Ridge...")

ridge_alphas = [
    0.001, 0.01, 0.1, 1.0, 10.0,
    100.0, 500.0, 1000.0,
]

for alpha in ridge_alphas:

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])

    model.fit(X_train, y_train)

    prediction = model.predict(X_validation)

    metrics = evaluate_regression(
        y_validation,
        prediction,
    )

    tuning_results.append({
        "Model": "Ridge",
        "Parameters": json.dumps({
            "alpha": alpha
        }),
        "Validation_MAE": metrics["MAE"],
        "Validation_RMSE": metrics["RMSE"],
        "Validation_R2": metrics["R2"],
    })


# ------------------------------------------------------------
# ELASTIC NET / LASSO
# ------------------------------------------------------------

print("Tuning ElasticNet...")

elastic_alphas = [
    0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0,
]

elastic_l1_ratios = [
    0.1, 0.3, 0.5, 0.7, 0.9, 1.0,
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
                random_state=RANDOM_STATE,
            )),
        ])

        model.fit(X_train, y_train)

        prediction = model.predict(X_validation)

        metrics = evaluate_regression(
            y_validation,
            prediction,
        )

        tuning_results.append({
            "Model": "ElasticNet",
            "Parameters": json.dumps({
                "alpha": alpha,
                "l1_ratio": l1_ratio,
            }),
            "Validation_MAE": metrics["MAE"],
            "Validation_RMSE": metrics["RMSE"],
            "Validation_R2": metrics["R2"],
        })


# ------------------------------------------------------------
# XGBOOST
# ------------------------------------------------------------

print("Tuning XGBoost...")

try:
    from xgboost import XGBRegressor

    xgb_parameters = []

    for max_depth in [2, 3, 4]:

        for learning_rate in [0.01, 0.03, 0.05]:

            for n_estimators in [200, 400, 600]:

                for reg_lambda in [1, 5, 10]:

                    xgb_parameters.append({
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                        "reg_lambda": reg_lambda,
                    })

    total = len(xgb_parameters)

    for i, params in enumerate(
        xgb_parameters,
        start=1
    ):

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                **params,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

        model.fit(X_train, y_train)

        prediction = model.predict(X_validation)

        metrics = evaluate_regression(
            y_validation,
            prediction,
        )

        tuning_results.append({
            "Model": "XGBoost",
            "Parameters": json.dumps(params),
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
        "WARNING: XGBoost is unavailable. "
        "Install it using: pip install xgboost"
    )


# ============================================================
# 6. SELECT BEST CONFIGURATIONS
# ============================================================

tuning_df = pd.DataFrame(tuning_results)

if tuning_df.empty:
    raise ValueError("No hyperparameter configurations were evaluated.")

tuning_df = tuning_df.sort_values(
    "Validation_RMSE"
).reset_index(drop=True)

tuning_df.to_csv(
    OUTPUT_DIR / "historical_tuning_results.csv",
    index=False,
)

print("\nBEST VALIDATION CONFIGURATIONS")
print("-" * 75)

best_configs = {}

for model_name in tuning_df["Model"].unique():

    subset = tuning_df[
        tuning_df["Model"] == model_name
    ]

    best = subset.iloc[0]

    parameters = json.loads(best["Parameters"])

    best_configs[model_name] = parameters

    print(f"\n{model_name}")
    print("Parameters:", parameters)
    print(
        f"Validation RMSE: "
        f"{best['Validation_RMSE']:.4f}"
    )

with open(
    OUTPUT_DIR / "selected_hyperparameters.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        best_configs,
        file,
        indent=4,
    )


# ============================================================
# 7. REFIT SELECTED MODELS ON ALL PRE-EVENT DATA
# ============================================================

print("\n" + "=" * 75)
print("FINAL PRE-EVENT MODEL FITTING")
print("=" * 75)

models = {}

# Ridge
ridge_parameters = best_configs["Ridge"]

models["Ridge"] = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(**ridge_parameters)),
])

# ElasticNet / Lasso
elastic_parameters = best_configs["ElasticNet"]

elastic_name = (
    "Lasso"
    if elastic_parameters["l1_ratio"] == 1.0
    else "ElasticNet"
)

models[elastic_name] = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", ElasticNet(
        **elastic_parameters,
        max_iter=30000,
        random_state=RANDOM_STATE,
    )),
])

# XGBoost
if "XGBoost" in best_configs:

    xgb_parameters = best_configs["XGBoost"]

    models["XGBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            **xgb_parameters,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

for name, model in models.items():

    print(f"\nFitting {name}...")

    model.fit(X_pre_event, y_pre_event)

print("\nAll selected models fitted.")


# ============================================================
# 8. GENERATE HISTORICAL EVENT FORECASTS
# ============================================================

print("\n" + "=" * 75)
print("GENERATING HISTORICAL DROUGHT FORECASTS")
print("=" * 75)

predictions = pd.DataFrame({
    "forecast_origin_month": (
        event["forecast_origin_month"].values
    ),
    "target_month": event["target_month"].values,
    "observed_swba": y_event.values,
})

for name, model in models.items():

    predictions[name] = model.predict(X_event)

# Persistence baseline.
predictions["Persistence"] = (
    event[PERSISTENCE_FEATURE].astype(float).values
)

# Seasonal climatology based exclusively on pre-event data.
pre_event_months = pre_event["target_month"].dt.month
event_months = event["target_month"].dt.month

seasonal_means = (
    pre_event.assign(
        calendar_month=pre_event_months
    )
    .groupby("calendar_month")[TARGET]
    .mean()
)

global_mean = y_pre_event.mean()

predictions["Seasonal Climatology"] = (
    event_months.map(seasonal_means)
    .fillna(global_mean)
    .values
)

# Global pre-event training mean.
predictions["Training-Mean Baseline"] = global_mean

predictions.to_csv(
    OUTPUT_DIR / "historical_event_predictions.csv",
    index=False,
)

print("\nHISTORICAL EVENT PREDICTIONS")
print("-" * 75)

print(
    predictions.to_string(
        index=False,
        formatters={
            "observed_swba": "{:.4f}".format,
        },
    )
)


# ============================================================
# 9. CONTINUOUS FORECAST PERFORMANCE
# ============================================================

print("\n" + "=" * 75)
print("CONTINUOUS FORECAST PERFORMANCE")
print("=" * 75)

model_names = [
    col for col in predictions.columns
    if col not in {
        "forecast_origin_month",
        "target_month",
        "observed_swba",
    }
]

continuous_results = []

for name in model_names:

    actual = predictions["observed_swba"].values
    predicted = predictions[name].values

    metrics = evaluate_regression(actual, predicted)

    bias = np.mean(predicted - actual)

    continuous_results.append({
        "Model": name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "Bias": bias,
    })

continuous_df = pd.DataFrame(continuous_results)

continuous_df = continuous_df.sort_values(
    "RMSE"
).reset_index(drop=True)

continuous_df.to_csv(
    OUTPUT_DIR / "historical_event_continuous_metrics.csv",
    index=False,
)

print(
    continuous_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "R2": "{:.4f}".format,
            "Bias": "{:.4f}".format,
        },
    )
)


# ============================================================
# 10. DROUGHT DETECTION PERFORMANCE
# ============================================================

print("\n" + "=" * 75)
print("DROUGHT DETECTION PERFORMANCE")
print("=" * 75)

detection_results = []

for threshold in DROUGHT_THRESHOLDS:

    actual_drought = (
        predictions["observed_swba"] <= threshold
    ).astype(int)

    for name in model_names:

        predicted_drought = (
            predictions[name] <= threshold
        ).astype(int)

        tp = int(
            ((actual_drought == 1) &
             (predicted_drought == 1)).sum()
        )

        fp = int(
            ((actual_drought == 0) &
             (predicted_drought == 1)).sum()
        )

        fn = int(
            ((actual_drought == 1) &
             (predicted_drought == 0)).sum()
        )

        tn = int(
            ((actual_drought == 0) &
             (predicted_drought == 0)).sum()
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0 else np.nan
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0 else np.nan
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (
                not np.isnan(precision)
                and not np.isnan(recall)
                and precision + recall > 0
            )
            else np.nan
        )

        detection_results.append({
            "Threshold": threshold,
            "Model": name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

detection_df = pd.DataFrame(detection_results)

detection_df.to_csv(
    OUTPUT_DIR / "historical_event_detection_metrics.csv",
    index=False,
)

print(
    detection_df.to_string(
        index=False,
        formatters={
            "Precision": lambda x: (
                f"{x:.3f}" if pd.notna(x) else "N/A"
            ),
            "Recall": lambda x: (
                f"{x:.3f}" if pd.notna(x) else "N/A"
            ),
            "F1": lambda x: (
                f"{x:.3f}" if pd.notna(x) else "N/A"
            ),
        },
    )
)


# ============================================================
# 11. SAVE EVENT SUMMARY
# ============================================================

summary = {
    "experiment": "Historical 2015-2016 drought event",
    "training_observations": int(len(training)),
    "validation_observations": int(len(validation)),
    "pre_event_fitting_observations": int(len(pre_event)),
    "event_observations": int(len(event)),
    "number_of_predictors": int(len(feature_columns)),
    "event_start": str(event["target_month"].min().date()),
    "event_end": str(event["target_month"].max().date()),
    "drought_thresholds": DROUGHT_THRESHOLDS,
    "model_selection_metric": "Validation RMSE",
    "models": list(models.keys()),
    "baseline_models": [
        "Persistence",
        "Seasonal Climatology",
        "Training-Mean Baseline",
    ],
    "retraining_during_event": False,
}

with open(
    OUTPUT_DIR / "historical_event_summary.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(summary, file, indent=4)

print("\n" + "=" * 75)
print("HISTORICAL DROUGHT EVALUATION COMPLETED")
print("=" * 75)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print("-", path.name)
