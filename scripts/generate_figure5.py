from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


# ============================================================
# FIGURE 5: OBSERVED VS FORECAST SWBA
# Independent test period: 2024–2025
# Data loaded from Notebook 05 canonical export
# ============================================================

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

cwd = Path.cwd()

if (cwd / "data").exists():
    PROJECT_ROOT = cwd
elif (cwd.parent / "data").exists():
    PROJECT_ROOT = cwd.parent
else:
    raise FileNotFoundError("Could not locate project root.")

OUTPUT_DIR = PROJECT_ROOT / "outputs"

forecasts_path = (
    OUTPUT_DIR / "independent_test_forecasts.csv"
)

if not forecasts_path.exists():
    raise FileNotFoundError(
        f"Missing evaluation export: {forecasts_path}"
    )


# ------------------------------------------------------------
# 2. LOAD CANONICAL INDEPENDENT-TEST DATA
# ------------------------------------------------------------

plot_data = pd.read_csv(
    forecasts_path,
    parse_dates=["forecast_date"]
)

required_columns = [
    "forecast_date",
    "observed",
    "ridge_forecast",
    "persistence",
    "climatology"
]

missing_columns = [
    column
    for column in required_columns
    if column not in plot_data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required forecast columns: {missing_columns}"
    )

plot_data = (
    plot_data[required_columns]
    .copy()
    .sort_values("forecast_date")
    .reset_index(drop=True)
)

if len(plot_data) != 24:
    raise ValueError(
        f"Expected 24 independent-test observations; "
        f"found {len(plot_data)}."
    )

if plot_data.isna().any().any():
    raise ValueError(
        "Missing values detected in independent-test data."
    )

if plot_data["forecast_date"].duplicated().any():
    raise ValueError(
        "Duplicate forecast dates detected."
    )


# ------------------------------------------------------------
# 3. IDENTIFY MAY 2024 FROM THE EXPORTED DATA
# ------------------------------------------------------------

may_2024 = plot_data.loc[
    plot_data["forecast_date"]
    == pd.Timestamp("2024-05-01")
]

if len(may_2024) != 1:
    raise ValueError(
        "Expected exactly one May 2024 observation."
    )

may_observed = may_2024["observed"].iloc[0]
may_forecast = may_2024["ridge_forecast"].iloc[0]
may_date = may_2024["forecast_date"].iloc[0]


# ------------------------------------------------------------
# 4. CREATE FIGURE
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 6.5))

ax.plot(
    plot_data["forecast_date"],
    plot_data["observed"],
    color="#1F2937",
    linewidth=2.2,
    marker="o",
    markersize=5,
    label="Observed SWBA",
    zorder=4
)

ax.plot(
    plot_data["forecast_date"],
    plot_data["ridge_forecast"],
    color="#23866A",
    linewidth=2,
    marker="s",
    markersize=4.5,
    label="Ridge forecast",
    zorder=3
)

ax.plot(
    plot_data["forecast_date"],
    plot_data["persistence"],
    color="#8A9DB0",
    linewidth=1.5,
    linestyle="--",
    label="Persistence baseline",
    zorder=2
)

ax.axhline(
    y=0,
    color="#B7791F",
    linewidth=1.5,
    linestyle=":",
    label="Climatological normal (SWBA = 0)"
)


# ------------------------------------------------------------
# 5. HIGHLIGHT MAY 2024 EXTREME EVENT
# ------------------------------------------------------------

ax.annotate(
    "May 2024\nExtreme observed anomaly",
    xy=(may_date, may_observed),
    xytext=(
        pd.Timestamp("2024-08-01"),
        may_observed - 0.65
    ),
    fontsize=9,
    fontweight="bold",
    color="#A33A32",
    arrowprops=dict(
        arrowstyle="->",
        color="#A33A32",
        linewidth=1.3
    )
)

ax.scatter(
    [may_date],
    [may_forecast],
    color="#23866A",
    s=65,
    zorder=5,
    edgecolor="white",
    linewidth=0.8
)


# ------------------------------------------------------------
# 6. AXIS FORMATTING
# ------------------------------------------------------------

start_date = plot_data["forecast_date"].min()
end_date = plot_data["forecast_date"].max()

ax.set_title(
    (
        "Observed and Forecast SWBA During "
        f"Independent Testing ({start_date.year}–{end_date.year})"
    ),
    fontsize=14,
    fontweight="bold",
    pad=16
)

ax.set_xlabel(
    "Forecast month",
    fontsize=11
)

ax.set_ylabel(
    "Standardized Water Balance Anomaly (SWBA)",
    fontsize=11
)

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=3)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.setp(
    ax.get_xticklabels(),
    rotation=35,
    ha="right"
)

ax.grid(
    axis="both",
    linestyle="--",
    alpha=0.25
)

ax.legend(
    loc="upper right",
    frameon=False,
    fontsize=9
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()


# ------------------------------------------------------------
# 7. SAVE FIGURE
# ------------------------------------------------------------

png_path = OUTPUT_DIR / "figure5_observed_vs_forecast.png"
pdf_path = OUTPUT_DIR / "figure5_observed_vs_forecast.pdf"

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nFigure 5 generated successfully!")
print("PNG:", png_path)
print("PDF:", pdf_path)

print("\nSource: independent_test_forecasts.csv")
print("Test observations:", len(plot_data))
print("Evaluation period:", start_date.date(), "to", end_date.date())
print("May 2024 observed SWBA:", may_observed)
print("May 2024 Ridge forecast:", may_forecast)
