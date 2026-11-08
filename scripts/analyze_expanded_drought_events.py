
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# HISTORICAL DRY-ANOMALY ANALYSIS
# Mnjoli W60E Catchment | 1981–2025
# Index: 3-month Standardized Water Balance Anomaly (SWBA)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "expanded"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVENTS_PATH = (
    OUTPUT_DIR
    / "historical_dry_anomaly_events_1981_2025.csv"
)

ANNUAL_PATH = (
    OUTPUT_DIR
    / "historical_dry_anomaly_annual_summary_1981_2025.csv"
)

SWBA_COLUMN = "swba_3month"

# Thresholds
MODERATE_THRESHOLD = -1.0
SEVERE_THRESHOLD = -2.0


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=["month_date"]
)

df = df.sort_values(
    "month_date"
).reset_index(drop=True)

df = df.dropna(
    subset=[SWBA_COLUMN]
).copy()

df["year"] = df["month_date"].dt.year
df["month"] = df["month_date"].dt.month

print("=" * 70)
print("HISTORICAL DRY-ANOMALY ANALYSIS")
print("=" * 70)

print(f"Valid monthly observations: {len(df):,}")
print(
    f"Period: {df['month_date'].min().strftime('%Y-%m')} "
    f"to {df['month_date'].max().strftime('%Y-%m')}"
)


# ============================================================
# 2. IDENTIFY DRY-ANOMALY MONTHS
# ============================================================

df["moderately_or_more_dry"] = (
    df[SWBA_COLUMN] <= MODERATE_THRESHOLD
)

df["severely_dry"] = (
    df[SWBA_COLUMN] <= SEVERE_THRESHOLD
)

print("\n" + "=" * 70)
print("DRY-ANOMALY FREQUENCY")
print("=" * 70)

moderate_count = df["moderately_or_more_dry"].sum()
severe_count = df["severely_dry"].sum()

print(
    f"SWBA <= {MODERATE_THRESHOLD}: "
    f"{moderate_count} months "
    f"({moderate_count / len(df) * 100:.2f}%)"
)

print(
    f"SWBA <= {SEVERE_THRESHOLD}: "
    f"{severe_count} months "
    f"({severe_count / len(df) * 100:.2f}%)"
)


# ============================================================
# 3. IDENTIFY CONSECUTIVE DRY-ANOMALY EVENTS
# ============================================================

# A new event begins when a dry month is not immediately
# preceded by another dry month.

df["dry_event_group"] = (
    df["moderately_or_more_dry"]
    != df["moderately_or_more_dry"].shift()
).cumsum()

dry_months = df.loc[
    df["moderately_or_more_dry"]
].copy()

events = (
    dry_months.groupby("dry_event_group")
    .agg(
        start_date=("month_date", "min"),
        end_date=("month_date", "max"),
        duration_months=("month_date", "count"),
        minimum_swba=(SWBA_COLUMN, "min"),
        mean_swba=(SWBA_COLUMN, "mean"),
        severe_months=("severely_dry", "sum")
    )
    .reset_index(drop=True)
)

events["start_year"] = (
    events["start_date"].dt.year
)

events["end_year"] = (
    events["end_date"].dt.year
)

events = events.sort_values(
    "start_date"
).reset_index(drop=True)


# ============================================================
# 4. SAVE EVENT-LEVEL RESULTS
# ============================================================

events.to_csv(
    EVENTS_PATH,
    index=False
)

print("\n" + "=" * 70)
print("CONSECUTIVE DRY-ANOMALY EVENTS")
print("=" * 70)

print(
    f"Number of events (SWBA <= {MODERATE_THRESHOLD}): "
    f"{len(events)}"
)

if not events.empty:

    print("\nLongest events:")
    print(
        events.sort_values(
            "duration_months",
            ascending=False
        )
        .head(10)
        [
            [
                "start_date",
                "end_date",
                "duration_months",
                "minimum_swba",
                "severe_months"
            ]
        ]
        .to_string(index=False)
    )

    print("\nMost intense events:")
    print(
        events.sort_values(
            "minimum_swba"
        )
        .head(10)
        [
            [
                "start_date",
                "end_date",
                "duration_months",
                "minimum_swba",
                "severe_months"
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# 5. ANNUAL DRY-ANOMALY SUMMARY
# ============================================================

annual = (
    df.groupby("year")
    .agg(
        valid_months=(SWBA_COLUMN, "count"),
        mean_swba=(SWBA_COLUMN, "mean"),
        minimum_swba=(SWBA_COLUMN, "min"),
        moderate_dry_months=("moderately_or_more_dry", "sum"),
        severe_dry_months=("severely_dry", "sum")
    )
    .reset_index()
)

annual["moderate_dry_percentage"] = (
    annual["moderate_dry_months"]
    / annual["valid_months"]
    * 100
)

annual["severe_dry_percentage"] = (
    annual["severe_dry_months"]
    / annual["valid_months"]
    * 100
)

annual.to_csv(
    ANNUAL_PATH,
    index=False
)


# ============================================================
# 6. IDENTIFY YEARS WITH MOST DRY-ANOMALY MONTHS
# ============================================================

print("\n" + "=" * 70)
print("YEARS WITH MOST MODERATE-OR-MORE DRY MONTHS")
print("=" * 70)

print(
    annual.sort_values(
        "moderate_dry_months",
        ascending=False
    )
    .head(15)
    [
        [
            "year",
            "moderate_dry_months",
            "severe_dry_months",
            "minimum_swba",
            "mean_swba"
        ]
    ]
    .to_string(index=False)
)


# ============================================================
# 7. OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OVERALL HISTORICAL SUMMARY")
print("=" * 70)

print(f"Valid observations: {len(df)}")
print(f"Dry-anomaly events: {len(events)}")

if not events.empty:

    print(
        "Mean event duration: "
        f"{events['duration_months'].mean():.2f} months"
    )

    print(
        "Longest event: "
        f"{events['duration_months'].max()} months"
    )

    print(
        "Most negative event SWBA: "
        f"{events['minimum_swba'].min():.4f}"
    )

print("\nEvent file saved to:")
print(EVENTS_PATH)

print("\nAnnual summary saved to:")
print(ANNUAL_PATH)

print("\nHISTORICAL DRY-ANOMALY ANALYSIS COMPLETE")
