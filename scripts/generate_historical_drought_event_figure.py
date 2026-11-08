
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "expanded"
    / "historical_2015_drought_event"
    / "historical_event_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "expanded"
    / "historical_2015_drought_event"
    / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Predictions file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# FLEXIBLE COLUMN DETECTION
# ============================================================

def normalise(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(candidates, required=True):
    lookup = {
        normalise(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = normalise(candidate)

        if key in lookup:
            return lookup[key]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return None


date_col = find_column([
    "target_month",
    "forecast_month",
    "date"
])

actual_col = find_column([
    "observed_swba",
    "observed_swba_3month",
    "actual_swba_3month",
    "observed",
    "actual",
    "target_swba_3month"
])

model_columns = {
    "Ridge": find_column([
        "Ridge",
        "Ridge_prediction",
        "ridge_pred"
    ]),
    "ElasticNet": find_column([
        "ElasticNet",
        "ElasticNet_prediction",
        "elasticnet_pred"
    ]),
    "XGBoost": find_column([
        "XGBoost",
        "XGBoost_prediction",
        "xgboost_pred"
    ]),
    "Persistence": find_column([
        "Persistence",
        "Persistence_prediction",
        "persistence_pred"
    ])
}


# ============================================================
# PREPARE DATA
# ============================================================

df[date_col] = pd.to_datetime(df[date_col])

df = df.sort_values(date_col).copy()

df[actual_col] = pd.to_numeric(
    df[actual_col],
    errors="coerce"
)

for column in model_columns.values():
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(
    subset=[date_col, actual_col]
)

if len(df) != 4:
    print(
        f"\nWARNING: Expected 4 event months, "
        f"but found {len(df)} rows."
    )

print("\nHistorical drought-event data:")
print(df[[date_col, actual_col] + list(model_columns.values())])


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Observed SWBA
ax.plot(
    df[date_col],
    df[actual_col],
    marker="o",
    linewidth=3,
    markersize=9,
    label="Observed SWBA",
    zorder=5
)

# Model predictions
for model_name, column in model_columns.items():
    ax.plot(
        df[date_col],
        df[column],
        marker="s",
        linewidth=2,
        markersize=6,
        linestyle="--",
        label=f"{model_name} prediction"
    )


# ============================================================
# DROUGHT THRESHOLDS
# ============================================================

ax.axhline(
    y=-1,
    linestyle=":",
    linewidth=2,
    label="Drought threshold (SWBA = -1)"
)

ax.axhline(
    y=-2,
    linestyle="-.",
    linewidth=2,
    label="Exceptional dryness threshold (SWBA = -2)"
)


# ============================================================
# EVENT ANNOTATIONS
# ============================================================

onset_date = pd.Timestamp("2015-11-01")

ax.axvline(
    onset_date,
    linestyle=":",
    linewidth=1.5
)

ax.annotate(
    "Drought onset",
    xy=(onset_date, -1.0),
    xytext=(8, 18),
    textcoords="offset points",
    fontsize=9,
    rotation=90,
    va="bottom"
)

exceptional_months = df[
    df[actual_col] <= -2
]

for _, row in exceptional_months.iterrows():
    ax.annotate(
        "Exceptional dryness",
        xy=(row[date_col], row[actual_col]),
        xytext=(0, -28),
        textcoords="offset points",
        ha="center",
        fontsize=8
    )


# ============================================================
# FORMATTING
# ============================================================

ax.set_title(
    "Historical Drought Event Evaluation: Mnjoli W60E Catchment\n"
    "Observed and Predicted Three-Month Standardized Water Balance Anomaly",
    fontsize=13,
    fontweight="bold",
    pad=16
)

ax.set_xlabel(
    "Target month",
    fontsize=11
)

ax.set_ylabel(
    "Standardized Water Balance Anomaly (SWBA)",
    fontsize=11
)

ax.set_xticks(df[date_col])

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.setp(
    ax.get_xticklabels(),
    rotation=25,
    ha="right"
)

ax.grid(
    True,
    linestyle=":",
    alpha=0.5
)

ax.legend(
    loc="best",
    fontsize=9,
    frameon=True
)

ax.margins(x=0.08)

plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

png_path = OUTPUT_DIR / "historical_2015_drought_event.png"
pdf_path = OUTPUT_DIR / "historical_2015_drought_event.pdf"

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.show()

print("\n" + "=" * 65)
print("HISTORICAL DROUGHT EVENT FIGURE GENERATED")
print("=" * 65)
print(f"PNG: {png_path}")
print(f"PDF: {pdf_path}")
