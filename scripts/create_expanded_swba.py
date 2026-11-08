
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CREATE EXPANDED STANDARDIZED WATER BALANCE ANOMALY (SWBA)
# Mnjoli W60E Catchment | 1981–2025
# Reference climatology: 1981–2010
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)

BASELINE_START = 1981
BASELINE_END = 2010

WATER_BALANCE_COLUMN = "water_balance_3month"
SWBA_COLUMN = "swba_3month"


# ============================================================
# 1. LOAD MONTHLY DATA
# ============================================================

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=["month_date"]
)

df = df.sort_values("month_date").reset_index(drop=True)

print("=" * 70)
print("EXPANDED SWBA CONSTRUCTION")
print("=" * 70)

print(f"Input records: {len(df):,}")
print(f"Date range: {df['month_date'].min()} to "
      f"{df['month_date'].max()}")


# ============================================================
# 2. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "month_date",
    "year",
    "month",
    WATER_BALANCE_COLUMN
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df["month_date"].duplicated().any():
    raise ValueError(
        "Duplicate monthly dates detected."
    )


# ============================================================
# 3. DEFINE THE REFERENCE CLIMATOLOGY
# ============================================================

baseline = df.loc[
    (df["year"] >= BASELINE_START)
    & (df["year"] <= BASELINE_END)
].copy()

print("\nReference climatology:")
print(f"{BASELINE_START}–{BASELINE_END}")
print(f"Baseline records: {len(baseline)}")


# ============================================================
# 4. CALCULATE CALENDAR-MONTH CLIMATOLOGY
# ============================================================

climatology = (
    baseline.groupby("month")[WATER_BALANCE_COLUMN]
    .agg(
        climatology_mean="mean",
        climatology_std="std",
        baseline_count="count"
    )
    .reset_index()
)

if climatology["climatology_mean"].isna().any():
    raise ValueError(
        "Missing climatological means detected."
    )

if climatology["climatology_std"].isna().any():
    raise ValueError(
        "Missing climatological standard deviations detected."
    )

if (climatology["climatology_std"] <= 0).any():
    raise ValueError(
        "Zero or negative climatological standard deviation."
    )


# ============================================================
# 5. MERGE CLIMATOLOGY WITH THE FULL HISTORICAL DATA
# ============================================================

df = df.merge(
    climatology,
    on="month",
    how="left",
    validate="many_to_one"
)


# ============================================================
# 6. CALCULATE SWBA
# ============================================================

df[SWBA_COLUMN] = (
    df[WATER_BALANCE_COLUMN]
    - df["climatology_mean"]
) / df["climatology_std"]


# ============================================================
# 7. CLASSIFY SWBA CONDITIONS
# ============================================================

conditions = [
    df[SWBA_COLUMN] <= -2,
    df[SWBA_COLUMN] <= -1,
    df[SWBA_COLUMN] < 0,
    df[SWBA_COLUMN] <= 1,
    df[SWBA_COLUMN] <= 2,
    df[SWBA_COLUMN] > 2
]

categories = [
    "Exceptionally Dry",
    "Severely Dry",
    "Moderately Dry",
    "Near Normal",
    "Moderately Wet",
    "Exceptionally Wet"
]

df["swba_category"] = np.select(
    conditions,
    categories,
    default="Unclassified"
)


# ============================================================
# 8. SAVE THE EXPANDED DATASET
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 9. REPORT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SWBA SUMMARY")
print("=" * 70)

print(
    df[SWBA_COLUMN]
    .describe()
    .round(4)
)

print("\nSWBA category counts:")
print(
    df["swba_category"]
    .value_counts(dropna=False)
)

print("\nMissing SWBA values:")
print(df[SWBA_COLUMN].isna().sum())

print("\nOutput saved to:")
print(OUTPUT_PATH)

print("\nSWBA CONSTRUCTION COMPLETE")
