from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# EXPANDING-WINDOW ROLLING-ORIGIN VALIDATION
# Models: Tuned XGBoost, Lasso (selected ElasticNet), Ridge,
#         Persistence
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "processed" / "expanded"
    / "mnjoli_expanded_forecasting_dataset_1981_2025.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "expanded"
    / "rolling_origin_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

df["target_month"] = pd.to_datetime(df["target_month"])

df = df.sort_values("target_month").reset_index(drop=True)

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

# Only observations through 2019 are used for rolling validation.
# The 2020–2025 test period remains untouched.
development = df[
    df["target_month"].dt.year <= 2019
].copy()

print("=" * 70)
print("ROLLING-ORIGIN VALIDATION")
print("=" * 70)

print(
    "Development period:",
    development["target_month"].min().date(),
    "to",
    development["target_month"].max().date()
)

print(f"Development observations: {len(development)}")
print(f"Predictor features: {len(FEATURES)}")

# ------------------------------------------------------------
# 2. Define expanding-window folds
#
# Each validation fold represents a four-year period.
# Training data expands forward chronologically.
# ------------------------------------------------------------

folds = [
    {
        "name": "Fold 1",
        "train_end": "2005-12-01",
        "val_start": "2006-01-01",
        "val_end": "2009-12-01",
    },
    {
        "name": "Fold 2",
        "train_end": "2009-12-01",
        "val_start": "2010-01-01",
        "val_end": "2013-12-01",
    },
    {
        "name": "Fold 3",
        "train_end": "2013-12-01",
        "val_start": "2014-01-01",
        "val_end": "2017-12-01",
    },
    {
        "name": "Fold 4",
        "train_end": "2015-12-01",
        "val_start": "2016-01-01",
        "val_end": "2019-12-01",
    },
]

# ------------------------------------------------------------
# 3. Define tuned models
# ------------------------------------------------------------

def create_models():

    models = {

        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),

        "ElasticNet_Lasso": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Lasso(
                alpha=0.01,
                max_iter=30000,
                random_state=42
            )),
        ]),
    }

    try:

        from xgboost import XGBRegressor

        models["Tuned_XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                max_depth=4,
                learning_rate=0.03,
                n_estimators=600,
                reg_lambda=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )),
        ])

    except ImportError:

        raise ImportError(
            "XGBoost is required. Install it with pip install xgboost"
        )

    return models


# ------------------------------------------------------------
# 4. Metric function
# ------------------------------------------------------------

def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),

        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),

        "R2": r2_score(actual, predicted),
    }


# ------------------------------------------------------------
# 5. Execute rolling-origin folds
# ------------------------------------------------------------

fold_results = []

for fold in folds:

    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    fold_train = development[
        development["target_month"] <= train_end
    ].copy()

    fold_val = development[
        development["target_month"].between(
            val_start,
            val_end
        )
    ].copy()

    if len(fold_train) == 0 or len(fold_val) == 0:

        print(f"Skipping {fold['name']}: insufficient observations.")
        continue

    X_train = fold_train[FEATURES]
    y_train = fold_train[TARGET]

    X_val = fold_val[FEATURES]
    y_val = fold_val[TARGET]

    print("\n" + "-" * 70)
    print(f"{fold['name']}")
    print(
        f"Training: {fold_train['target_month'].min().date()} "
        f"to {fold_train['target_month'].max().date()} "
        f"| n = {len(fold_train)}"
    )
    print(
        f"Validation: {fold_val['target_month'].min().date()} "
        f"to {fold_val['target_month'].max().date()} "
        f"| n = {len(fold_val)}"
    )

    models = create_models()

    # --------------------------------------------------------
    # ML models
    # --------------------------------------------------------

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        metrics = calculate_metrics(
            y_val,
            predictions
        )

        fold_results.append({
            "Fold": fold["name"],
            "Model": name,
            "Train_n": len(fold_train),
            "Validation_n": len(fold_val),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
        })

        print(
            f"{name:22s} "
            f"RMSE = {metrics['RMSE']:.4f} | "
            f"MAE = {metrics['MAE']:.4f} | "
            f"R2 = {metrics['R2']:.4f}"
        )

    # --------------------------------------------------------
    # Persistence baseline
    # --------------------------------------------------------

    persistence_feature = "swba_3month_lag_0"

    persistence_predictions = (
        X_val[persistence_feature].to_numpy()
    )

    metrics = calculate_metrics(
        y_val,
        persistence_predictions
    )

    fold_results.append({
        "Fold": fold["name"],
        "Model": "Persistence",
        "Train_n": len(fold_train),
        "Validation_n": len(fold_val),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    })

    print(
        f"{'Persistence':22s} "
        f"RMSE = {metrics['RMSE']:.4f} | "
        f"MAE = {metrics['MAE']:.4f} | "
        f"R2 = {metrics['R2']:.4f}"
    )

# ------------------------------------------------------------
# 6. Save fold-level results
# ------------------------------------------------------------

fold_df = pd.DataFrame(fold_results)

fold_file = (
    OUTPUT_DIR
    / "rolling_origin_fold_results.csv"
)

fold_df.to_csv(
    fold_file,
    index=False
)

# ------------------------------------------------------------
# 7. Calculate mean and standard deviation across folds
# ------------------------------------------------------------

summary_df = (
    fold_df.groupby("Model")
    .agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
    )
    .reset_index()
)

summary_df = summary_df.sort_values(
    "RMSE_mean"
).reset_index(drop=True)

summary_file = (
    OUTPUT_DIR
    / "rolling_origin_validation_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

# ------------------------------------------------------------
# 8. Display summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ROLLING-ORIGIN SUMMARY: MEAN ± STANDARD DEVIATION")
print("=" * 70)

for _, row in summary_df.iterrows():

    print(f"\n{row['Model']}")

    print(
        f"MAE:  {row['MAE_mean']:.4f} "
        f"+/- {row['MAE_std']:.4f}"
    )

    print(
        f"RMSE: {row['RMSE_mean']:.4f} "
        f"+/- {row['RMSE_std']:.4f}"
    )

    print(
        f"R2:   {row['R2_mean']:.4f} "
        f"+/- {row['R2_std']:.4f}"
    )

print("\n" + "=" * 70)
print("ROLLING-ORIGIN VALIDATION COMPLETE")
print("=" * 70)

print("\nFold results saved to:")
print(fold_file)

print("\nSummary saved to:")
print(summary_file)
