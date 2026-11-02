
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


# ============================================================
# FIGURE 5: OBSERVED VS FORECAST SWBA
# Independent test period: 2024–2025
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. OFFICIAL INDEPENDENT-TEST DATA
# ------------------------------------------------------------

dates = pd.date_range(
    start="2024-01-01",
    periods=24,
    freq="MS"
)

observed = [
    0.6592, 0.1948, 0.9950, 2.0794,
    7.0614, 1.3644, -0.6764, -1.0491,
    -0.7537, 0.4731, -0.0304, -0.1491,
    0.1455, 0.6178, 0.6282, 1.1314,
    1.1437, 1.0692, -0.7045, -0.6790,
    1.0983, 0.8245, 1.9902, 1.7403
]

ridge_forecast = [
    0.6808, 1.2087, -0.7691, 1.2341,
    1.9365, -0.1616, -0.1716, -0.2326,
    -0.0850, 0.0656, 0.3435, -0.3644,
    -0.5309, 0.1170, 0.1781, 0.7071,
    0.3758, 1.4126, 0.2256, -0.1311,
    -0.1096, -0.5750, -0.1786, 1.3403
]

persistence = [
    1.1365, 0.6592, 0.1948, 0.9950,
    2.0794, 7.0614, 1.3644, -0.6764,
    -1.0491, -0.7537, 0.4731, -0.0304,
    -0.1491, 0.1455, 0.6178, 0.6282,
    1.1314, 1.1437, 1.0692, -0.7045,
    -0.6790, 1.0983, 0.8245, 1.9902
]


# ------------------------------------------------------------
# 3. CREATE DATAFRAME
# ------------------------------------------------------------

plot_data = pd.DataFrame({
    "Date": dates,
    "Observed SWBA": observed,
    "Ridge forecast": ridge_forecast,
    "Persistence": persistence
})


# ------------------------------------------------------------
# 4. CREATE FIGURE
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 6.5))

ax.plot(
    plot_data["Date"],
    plot_data["Observed SWBA"],
    color="#1F2937",
    linewidth=2.2,
    marker="o",
    markersize=5,
    label="Observed SWBA",
    zorder=4
)

ax.plot(
    plot_data["Date"],
    plot_data["Ridge forecast"],
    color="#23866A",
    linewidth=2,
    marker="s",
    markersize=4.5,
    label="Ridge forecast",
    zorder=3
)

ax.plot(
    plot_data["Date"],
    plot_data["Persistence"],
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

may_2024 = pd.Timestamp("2024-05-01")

may_observed = 7.0614
may_forecast = 1.9365

ax.annotate(
    "May 2024\nExtreme observed anomaly",
    xy=(may_2024, may_observed),
    xytext=(pd.Timestamp("2024-08-01"), 6.4),
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
    [may_2024],
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

ax.set_title(
    "Observed and Forecast SWBA During Independent Testing",
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
