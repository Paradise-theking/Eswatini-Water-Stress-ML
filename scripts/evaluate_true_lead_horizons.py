from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    ROOT / "data" / "processed" / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "expanded" / "horizon_comparison"

TARGET = "swba_3month"
BASE_PREDICTORS = [
    "precipitation_mm", "pet_mm", "water_balance_mm",
    "temperature_mean_c", "temperature_min_c", "temperature_max_c",
    "dewpoint_c", "surface_pressure", "u_component_of_wind_10m",
    "v_component_of_wind_10m", "surface_solar_radiation_downwards_sum",
    "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
    "runoff_sum", "surface_runoff_sum", "water_balance_3month",
    "water_balance_6month", "swba_3month",
]
LAGS = [0, 1, 3, 6, 12]
LEADS = [1, 2, 3]
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2025-12-01")


def build_horizon_dataset(source: pd.DataFrame, lead_months: int) -> tuple[pd.DataFrame, list[str]]:
    """Build an origin-t feature matrix for a target at t + lead_months."""
    if lead_months < 1:
        raise ValueError("lead_months must be at least 1")

    df = source.copy()
    df["month_date"] = pd.to_datetime(df["month_date"])
    df = df.sort_values("month_date").reset_index(drop=True)

    required = ["month_date", TARGET, *BASE_PREDICTORS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Monthly SWBA input is missing columns: {missing}")

    if df["month_date"].duplicated().any():
        raise ValueError("Monthly SWBA input contains duplicate months.")

    frame = pd.DataFrame(index=df.index)
    frame["forecast_origin_month"] = df["month_date"]
    frame["target_month"] = df["month_date"].shift(-lead_months)
    frame["target_swba_3month"] = df[TARGET].shift(-lead_months)

    feature_columns = []
    for variable in BASE_PREDICTORS:
        for lag in LAGS:
            name = f"{variable}_lag_{lag}"
            frame[name] = df[variable].shift(lag)
            feature_columns.append(name)

    target_month_number = frame["target_month"].dt.month
    frame["target_month_sin"] = np.sin(2 * np.pi * target_month_number / 12)
    frame["target_month_cos"] = np.cos(2 * np.pi * target_month_number / 12)
    feature_columns.extend(["target_month_sin", "target_month_cos"])

    frame = frame.dropna(
        subset=["forecast_origin_month", "target_month", "target_swba_3month", *feature_columns]
    ).copy()

    expected_target = (
        frame["forecast_origin_month"].dt.to_period("M") + lead_months
    )
    actual_target = frame["target_month"].dt.to_period("M")
    if not (expected_target == actual_target).all():
        raise AssertionError("Forecast origin and target month are not aligned.")

    # The target month is the month in which the rolling 3-month SWBA is
    # observed. Inputs are restricted to the origin month and earlier.
    frame["input_cutoff_month"] = (
        frame["forecast_origin_month"] + pd.offsets.MonthEnd(0)
    ).dt.strftime("%Y-%m-%d")
    frame["lead_months"] = lead_months
    return frame.reset_index(drop=True), feature_columns


def build_models():
    models = {
        "Lasso": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=50000)),
        ]),
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
    }

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
        print("XGBoost not installed; evaluating Lasso and Ridge only.")

    return models


def evaluate_lead(frame: pd.DataFrame, features: list[str], lead_months: int):
    test = frame[
        frame["target_month"].between(TEST_START, TEST_END)
    ].copy()

    if test.empty:
        raise ValueError(f"Lead {lead_months}: test split is empty.")

    # Freeze the candidate at the earliest test forecast origin. Every
    # training target must already have completed by that issue origin.
    first_issue_origin = test["forecast_origin_month"].min()
    development = frame[
        frame["target_month"] <= first_issue_origin
    ].copy()

    if development.empty:
        raise ValueError(f"Lead {lead_months}: no training labels were available at first test issue origin.")

    if development["target_month"].max() > first_issue_origin:
        raise AssertionError("Training labels extend beyond the first test issue origin.")

    x_dev = development[features]
    y_dev = development["target_swba_3month"].astype(float)
    x_test = test[features]
    y_test = test["target_swba_3month"].astype(float)

    predictions = test[
        ["forecast_origin_month", "input_cutoff_month", "target_month",
         "lead_months", "target_swba_3month"]
    ].rename(columns={"target_swba_3month": "observed_swba"}).copy()
    predictions["training_target_cutoff"] = development["target_month"].max().strftime("%Y-%m")
    predictions["first_test_issue_origin"] = first_issue_origin.strftime("%Y-%m")
    predictions["structural_lead_valid"] = (
        predictions["target_month"] > predictions["forecast_origin_month"]
    )
    predictions["source_publication_time_verified"] = False

    prediction_series = {}
    for name, model in build_models().items():
        model.fit(x_dev, y_dev)
        prediction_series[name] = model.predict(x_test)

    # Persistence carries the origin-month SWBA forward to the target.
    prediction_series["Persistence"] = test["swba_3month_lag_0"].to_numpy(dtype=float)

    seasonal_means = (
        development.assign(target_calendar_month=development["target_month"].dt.month)
        .groupby("target_calendar_month")["target_swba_3month"]
        .mean()
    )
    global_mean = float(y_dev.mean())
    prediction_series["Seasonal Climatology"] = (
        test["target_month"].dt.month.map(seasonal_means).fillna(global_mean).to_numpy()
    )
    prediction_series["Training-Mean Baseline"] = np.full(len(test), global_mean, dtype=float)

    metrics = []
    for name, predicted in prediction_series.items():
        predictions[name] = np.asarray(predicted, dtype=float)
        metrics.append({
            "lead_months": lead_months,
            "model": name,
            "development_n": len(development),
            "training_target_cutoff": development["target_month"].max().strftime("%Y-%m"),
            "first_test_issue_origin": first_issue_origin.strftime("%Y-%m"),
            "test_n": len(test),
            "source_publication_time_verified": False,
            "test_start": test["target_month"].min().strftime("%Y-%m"),
            "test_end": test["target_month"].max().strftime("%Y-%m"),
            "MAE": mean_absolute_error(y_test, predicted),
            "RMSE": np.sqrt(mean_squared_error(y_test, predicted)),
            "R2": r2_score(y_test, predicted),
        })

    return predictions, metrics


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Required local input is missing: {INPUT_FILE}. "
            "Run the expanded SWBA preparation pipeline first."
        )

    source = pd.read_csv(INPUT_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prediction_frames = []
    metric_rows = []
    for lead in LEADS:
        frame, features = build_horizon_dataset(source, lead)
        predictions, metrics = evaluate_lead(frame, features, lead)
        prediction_frames.append(predictions)
        metric_rows.extend(metrics)
        print(f"Lead {lead}: development={metrics[0]['development_n']}, "
              f"test={metrics[0]['test_n']}")

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_metrics = pd.DataFrame(metric_rows).sort_values(
        ["lead_months", "RMSE"]
    )

    all_predictions.to_csv(
        OUTPUT_DIR / "horizon_test_predictions.csv", index=False
    )
    all_metrics.to_csv(
        OUTPUT_DIR / "horizon_test_metrics.csv", index=False
    )

    print("\nHORIZON COMPARISON")
    print(all_metrics.round(4).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
