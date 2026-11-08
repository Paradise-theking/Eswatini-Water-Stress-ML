
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 1. PROJECT PATHS
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
    / "outputs"
    / "expanded"
    / "historical_drought_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

THRESHOLDS = {
    "drought": -1.0,
    "severe_drought": -1.5,
    "exceptional_dryness": -2.0,
}

DATE_COLUMN = "month_date"
SWBA_COLUMN = "swba_3month"


# ============================================================
# 3. LOAD AND VALIDATE DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    DATE_COLUMN,
    SWBA_COLUMN,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN],
    errors="coerce"
)

df[SWBA_COLUMN] = pd.to_numeric(
    df[SWBA_COLUMN],
    errors="coerce"
)

df = df.dropna(subset=[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

if df[DATE_COLUMN].duplicated().any():
    raise ValueError(
        "Duplicate monthly dates detected. "
        "Resolve these before conducting the audit."
    )

df["year"] = df[DATE_COLUMN].dt.year
df["month"] = df[DATE_COLUMN].dt.month

valid = df[SWBA_COLUMN].notna()

print("=" * 70)
print("HISTORICAL DROUGHT EPISODE AUDIT")
print("=" * 70)

print(f"Total monthly records: {len(df)}")
print(f"Valid SWBA observations: {valid.sum()}")
print(f"Missing SWBA observations: {df[SWBA_COLUMN].isna().sum()}")
print(
    f"Date coverage: {df[DATE_COLUMN].min().date()} "
    f"to {df[DATE_COLUMN].max().date()}"
)


# ============================================================
# 4. IDENTIFY CONSECUTIVE DROUGHT EPISODES
# ============================================================

def identify_episodes(data, threshold, threshold_name):

    records = []

    drought_mask = (
        data[SWBA_COLUMN] <= threshold
    )

    # An episode continues only when the next valid
    # observation is exactly one calendar month later.
    episode_id = 0
    current_rows = []

    def save_episode(rows, current_id):

        if not rows:
            return

        values = np.array([
            row[SWBA_COLUMN]
            for row in rows
        ])

        dates = pd.to_datetime([
            row[DATE_COLUMN]
            for row in rows
        ])

        duration = len(rows)

        cumulative_deficit = np.maximum(
            threshold - values,
            0
        ).sum()

        records.append({
            "threshold_name": threshold_name,
            "threshold": threshold,
            "episode_id": current_id,
            "start_date": dates.min(),
            "end_date": dates.max(),
            "duration_months": duration,
            "minimum_swba": values.min(),
            "maximum_swba": values.max(),
            "mean_swba": values.mean(),
            "cumulative_deficit": cumulative_deficit,
            "months_below_threshold": duration,
        })

    previous_date = None

    for _, row in data.iterrows():

        current_date = row[DATE_COLUMN]
        current_value = row[SWBA_COLUMN]

        is_valid = pd.notna(current_value)

        consecutive = False

        if previous_date is not None:
            expected_next = (
                previous_date
                + pd.DateOffset(months=1)
            )

            consecutive = (
                current_date == expected_next
            )

        if (
            not is_valid
            or not drought_mask.loc[row.name]
        ):

            save_episode(
                current_rows,
                episode_id
            )

            current_rows = []
            previous_date = current_date
            continue

        if current_rows and not consecutive:

            save_episode(
                current_rows,
                episode_id
            )

            current_rows = []

        if not current_rows:
            episode_id += 1

        current_rows.append(row)

        previous_date = current_date

    save_episode(
        current_rows,
        episode_id
    )

    return pd.DataFrame(records)


# ============================================================
# 5. GENERATE EPISODE CATALOGUES
# ============================================================

all_episode_tables = []

for threshold_name, threshold in THRESHOLDS.items():

    episodes = identify_episodes(
        df,
        threshold,
        threshold_name
    )

    all_episode_tables.append(episodes)

    print("\n" + "-" * 65)
    print(
        f"{threshold_name.upper()} "
        f"(SWBA <= {threshold})"
    )
    print("-" * 65)

    if episodes.empty:
        print("No episodes identified.")
        continue

    print(f"Number of episodes: {len(episodes)}")
    print(
        "Total drought months:",
        episodes["duration_months"].sum()
    )

    print(
        "\nLongest episode:",
        episodes.loc[
            episodes["duration_months"].idxmax(),
            "duration_months"
        ],
        "months"
    )

    print(
        "\nMost severe episode:",
        episodes.loc[
            episodes["minimum_swba"].idxmin(),
            "start_date"
        ].strftime("%Y-%m"),
        "to",
        episodes.loc[
            episodes["minimum_swba"].idxmin(),
            "end_date"
        ].strftime("%Y-%m")
    )

    print(
        "Minimum SWBA:",
        episodes["minimum_swba"].min()
    )

    output_name = (
        f"drought_episodes_{threshold_name}.csv"
    )

    episodes.to_csv(
        OUTPUT_DIR / output_name,
        index=False
    )


# Combined episode catalogue
nonempty_tables = [
    table
    for table in all_episode_tables
    if not table.empty
]

if nonempty_tables:

    combined_episodes = pd.concat(
        nonempty_tables,
        ignore_index=True
    )

else:

    combined_episodes = pd.DataFrame()

combined_episodes.to_csv(
    OUTPUT_DIR / "all_historical_drought_episodes.csv",
    index=False
)


# ============================================================
# 6. MONTHLY DROUGHT CLASSIFICATION
# ============================================================

monthly = df[
    [
        DATE_COLUMN,
        SWBA_COLUMN,
        "year",
        "month"
    ]
].copy()

monthly["drought"] = (
    monthly[SWBA_COLUMN] <= -1.0
)

monthly["severe_drought"] = (
    monthly[SWBA_COLUMN] <= -1.5
)

monthly["exceptional_dryness"] = (
    monthly[SWBA_COLUMN] <= -2.0
)

monthly["drought_severity_class"] = np.select(
    [
        monthly[SWBA_COLUMN] <= -2.0,
        monthly[SWBA_COLUMN] <= -1.5,
        monthly[SWBA_COLUMN] <= -1.0,
    ],
    [
        "Exceptional Dryness",
        "Severe Drought",
        "Drought",
    ],
    default="Not Classified as Drought"
)

monthly.to_csv(
    OUTPUT_DIR / "monthly_drought_classification.csv",
    index=False
)


# ============================================================
# 7. ANNUAL DROUGHT SUMMARY
# ============================================================

annual = monthly.groupby("year").agg(
    valid_swba_months=(
        SWBA_COLUMN,
        "count"
    ),
    drought_months=(
        "drought",
        "sum"
    ),
    severe_drought_months=(
        "severe_drought",
        "sum"
    ),
    exceptional_dry_months=(
        "exceptional_dryness",
        "sum"
    ),
    minimum_swba=(
        SWBA_COLUMN,
        "min"
    ),
    mean_swba=(
        SWBA_COLUMN,
        "mean"
    )
).reset_index()

annual.to_csv(
    OUTPUT_DIR / "annual_drought_summary.csv",
    index=False
)


# ============================================================
# 8. HISTORICAL DROUGHT TIMELINE FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(
    monthly[DATE_COLUMN],
    monthly[SWBA_COLUMN],
    linewidth=1.3,
    label="Observed three-month SWBA"
)

# Reference thresholds
ax.axhline(
    -1,
    linestyle="--",
    linewidth=1.5,
    label="Drought threshold (-1)"
)

ax.axhline(
    -1.5,
    linestyle=":",
    linewidth=1.5,
    label="Severe drought threshold (-1.5)"
)

ax.axhline(
    -2,
    linestyle="-.",
    linewidth=1.5,
    label="Exceptional dryness threshold (-2)"
)

ax.axhline(
    0,
    linewidth=0.8,
    alpha=0.5
)

# Shade drought months
valid_drought = monthly[
    monthly["drought"]
]

for _, row in valid_drought.iterrows():

    ax.axvspan(
        row[DATE_COLUMN],
        row[DATE_COLUMN] + pd.DateOffset(months=1),
        alpha=0.10
    )

ax.set_title(
    "Historical Hydroclimatic Drought Conditions "
    "in the Mnjoli W60E Catchment (1981–2025)",
    fontsize=14,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "Year",
    fontsize=11
)

ax.set_ylabel(
    "Standardized Water Balance Anomaly (SWBA)",
    fontsize=11
)

ax.xaxis.set_major_locator(
    mdates.YearLocator(5)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y")
)

ax.grid(
    True,
    linestyle=":",
    alpha=0.4
)

ax.legend(
    loc="lower right",
    fontsize=9
)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "historical_drought_timeline.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "historical_drought_timeline.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HISTORICAL DROUGHT AUDIT COMPLETED")
print("=" * 70)

for threshold_name, threshold in THRESHOLDS.items():

    table = identify_episodes(
        df,
        threshold,
        threshold_name
    )

    print(
        f"{threshold_name}: "
        f"{len(table)} episodes, "
        f"{table['duration_months'].sum() if not table.empty else 0} "
        f"months"
    )

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(" -", path.name)
