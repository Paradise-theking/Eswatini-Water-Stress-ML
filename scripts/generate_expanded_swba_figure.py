
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# HISTORICAL 3-MONTH SWBA
# Mnjoli W60E Catchment | 1981–2025
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "expanded"
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    FIGURE_DIR
    / "historical_3month_swba_1981_2025.png"
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=["month_date"]
)

df = (
    df.sort_values("month_date")
    .dropna(subset=["swba_3month"])
    .reset_index(drop=True)
)

dates = df["month_date"]
swba = df["swba_3month"]


# ============================================================
# 2. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 7),
    dpi=300
)

# Historical normal
ax.axhline(
    0,
    color="black",
    linewidth=1.1,
    linestyle="-",
    zorder=2
)

# Moderate dry-anomaly threshold
ax.axhline(
    -1,
    color="#D97706",
    linewidth=1.4,
    linestyle="--",
    zorder=2
)

# Severe dry-anomaly threshold
ax.axhline(
    -2,
    color="#B91C1C",
    linewidth=1.8,
    linestyle=(0, (2, 2)),
    zorder=3
)

# Shade moderate-or-more dry-anomaly regions
ax.fill_between(
    dates,
    swba,
    -1,
    where=(swba <= -1),
    interpolate=True,
    color="#D97706",
    alpha=0.18,
    label="Dry-anomaly periods (SWBA ≤ −1)",
    zorder=1
)

# Monthly SWBA series
ax.plot(
    dates,
    swba,
    color="#2878B8",
    linewidth=0.9,
    label="Monthly 3-month SWBA",
    zorder=4
)

# Highlight severe dry-anomaly observations
severe = swba <= -2

ax.scatter(
    dates[severe],
    swba[severe],
    color="#B91C1C",
    s=22,
    marker="o",
    label="Severe dry anomalies (SWBA ≤ −2)",
    zorder=5
)


# ============================================================
# 3. LABELS AND FORMATTING
# ============================================================

ax.set_title(
    "Historical 3-Month Standardized Water Balance Anomaly "
    "(SWBA), 1981–2025\n"
    "Mnjoli W60E Catchment, Eswatini",
    fontsize=15,
    pad=14
)

ax.set_xlabel(
    "Year",
    fontsize=12
)

ax.set_ylabel(
    "Standardized Water Balance Anomaly (SWBA)",
    fontsize=12
)

ax.set_xlim(
    dates.min(),
    dates.max()
)

ax.grid(
    axis="both",
    linestyle=":",
    linewidth=0.6,
    alpha=0.35
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="lower left",
    fontsize=9,
    frameon=True,
    framealpha=0.95
)

fig.tight_layout()


# ============================================================
# 4. SAVE FIGURE
# ============================================================

fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 5. REPORT
# ============================================================

print("=" * 65)
print("UPDATED SWBA FIGURE GENERATED")
print("=" * 65)

print(f"Observations plotted: {len(df)}")
print(f"Dry-anomaly months: {(swba <= -1).sum()}")
print(f"Severe dry-anomaly months: {(swba <= -2).sum()}")

print(f"\nSaved to:\n{OUTPUT_PATH}")
