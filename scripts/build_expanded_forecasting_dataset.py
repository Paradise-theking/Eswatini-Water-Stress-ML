from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# EXPANDED FORECASTING DATASET: 1981–2025
# Target: Next month's 3-month SWBA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "mnjoli_expanded_forecasting_dataset_1981_2025.csv"
)

# ------------------------------------------------------------
# 1. Load and validate data
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["month_date"] = pd.to_datetime(df["month_date"])

df = (
    df.sort_values("month_date")
    .reset_index(drop=True)
)

if "swba_3month" not in df.columns:
    raise ValueError("Required target column 'swba_3month' is missing.")

print("=" * 70)
print("EXPANDED FORECASTING DATASET")
print("=" * 70)

print(f"Source file: {INPUT_FILE}")
print(f"Original observations: {len(df)}")
print(f"Date range: {df['month_date'].min()} to {df['month_date'].max()}")

# ------------------------------------------------------------
# 2. Define predictor variables
# ------------------------------------------------------------

base_predictors = [
    "precipitation_mm",
    "pet_mm",
    "water_balance_mm",
    "temperature_mean_c",
    "temperature_min_c",
    "temperature_max_c",
    "dewpoint_c",
    "surface_pressure",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_sum",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "runoff_sum",
    "surface_runoff_sum",
    "water_balance_3month",
    "water_balance_6month",
    "swba_3month",
]

missing_columns = [
    col for col in base_predictors
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing predictor columns: {missing_columns}"
    )

# ------------------------------------------------------------
# 3. Construct forecasting features
#
# Forecast origin = month t
# Target = SWBA at month t+1
#
# Lag 0 = information available at month t
# Lag 1 = information available at month t-1
# ------------------------------------------------------------

data = pd.DataFrame(index=df.index)

data["forecast_origin_month"] = df["month_date"]

data["target_month"] = df["month_date"].shift(-1)

data["target_swba_3month"] = (
    df["swba_3month"].shift(-1)
)

# Current and lagged predictor information
lags = [0, 1, 3, 6, 12]

for variable in base_predictors:

    for lag in lags:

        feature_name = (
            f"{variable}_lag_{lag}"
        )

        data[feature_name] = df[variable].shift(lag)

# ------------------------------------------------------------
# 4. Add known seasonal information
#
# The calendar month of the forecast target is known in advance.
# Encode it cyclically to preserve seasonal continuity.
# ------------------------------------------------------------

target_month_number = (
    data["target_month"].dt.month
)

data["target_month_sin"] = np.sin(
    2 * np.pi * target_month_number / 12
)

data["target_month_cos"] = np.cos(
    2 * np.pi * target_month_number / 12
)

# ------------------------------------------------------------
# 5. Assign chronological data partitions
# ------------------------------------------------------------

target_year = data["target_month"].dt.year

data["dataset_split"] = np.select(
    [
        target_year <= 2015,
        target_year.between(2016, 2019),
        target_year.between(2020, 2025),
    ],
    [
        "train",
        "validation",
        "test",
    ],
    default="exclude",
)

# ------------------------------------------------------------
# 6. Remove rows with unavailable target/features
#
# No imputation is performed at this stage.
# ------------------------------------------------------------

feature_columns = [
    col for col in data.columns
    if col not in [
        "forecast_origin_month",
        "target_month",
        "target_swba_3month",
        "dataset_split",
    ]
]

before_filtering = len(data)

data = data.dropna(
    subset=[
        "target_swba_3month",
        "target_month",
    ] + feature_columns
).copy()

data = data[
    data["dataset_split"] != "exclude"
].copy()

data = data.reset_index(drop=True)

# ------------------------------------------------------------
# 7. Save dataset
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

data.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# 8. Audit and report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET AUDIT")
print("=" * 70)

print(f"Rows before filtering: {before_filtering}")
print(f"Rows after filtering:  {len(data)}")
print(f"Predictor features:    {len(feature_columns)}")

print("\nSplit counts:")

print(
    data["dataset_split"]
    .value_counts()
    .reindex(
        ["train", "validation", "test"],
        fill_value=0
    )
)

print("\nTarget period by split:")

for split in ["train", "validation", "test"]:

    subset = data[
        data["dataset_split"] == split
    ]

    if len(subset) > 0:

        print(
            f"{split.upper():12} "
            f"{subset['target_month'].min().date()} "
            f"to "
            f"{subset['target_month'].max().date()} "
            f"| n = {len(subset)}"
        )

print("\nMissing values remaining:")

print(
    int(data[feature_columns].isna().sum().sum())
    + int(data["target_swba_3month"].isna().sum())
)

print("\nFirst 5 records:")

print(
    data[
        [
            "forecast_origin_month",
            "target_month",
            "target_swba_3month",
            "dataset_split",
        ]
    ].head().to_string(index=False)
)

print("\nSaved successfully to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)
