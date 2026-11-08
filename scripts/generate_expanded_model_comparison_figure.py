from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    PROJECT_ROOT / "outputs" / "expanded"
    / "final_evaluation" / "final_test_metrics.csv"
)

OUTPUT_DIR = RESULTS_PATH.parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(RESULTS_PATH)

label_map = {
    "Lasso": "Lasso",
    "Ridge": "Ridge",
    "Tuned XGBoost": "XGBoost",
    "Persistence": "Persistence",
    "Seasonal Climatology": "Seasonal Climatology",
    "Training-Mean Baseline": "Training-Mean Baseline",
}

df["Display_Model"] = df["Model"].map(label_map)

df = df.sort_values(
    "Test_RMSE",
    ascending=True
).reset_index(drop=True)

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(20, 8),
    gridspec_kw={"wspace": 0.48}
)

metrics = [
    ("Test_MAE", "Mean Absolute Error (MAE)", "Lower is better"),
    ("Test_RMSE", "Root Mean Squared Error (RMSE)", "Lower is better"),
    ("Test_R2", "Coefficient of Determination ($R^2$)", "Higher is better"),
]

models = df["Display_Model"].tolist()

# ============================================================
# DRAW PANELS
# ============================================================

for ax, (metric, title, direction) in zip(axes, metrics):

    values = df[metric].values

    bars = ax.barh(
        models,
        values,
        height=0.60
    )

    ax.set_title(
        title,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel(direction, labelpad=8)
    ax.tick_params(axis="y", pad=8)

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )

    ax.set_axisbelow(True)
    ax.invert_yaxis()

    # Keep labels inside bars for positive bars where possible
    for bar, value in zip(bars, values):

        width = bar.get_width()

        if metric == "Test_R2" and value < 0:
            ax.text(
                width - 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="right",
                fontsize=8
            )

        elif width > 0.15:
            ax.text(
                width - 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="right",
                fontsize=8,
                fontweight="bold"
            )

        else:
            ax.text(
                width + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=8
            )

# ============================================================
# AXIS LIMITS
# ============================================================

# MAE
axes[0].set_xlim(
    0,
    df["Test_MAE"].max() * 1.20
)

# RMSE
axes[1].set_xlim(
    0,
    df["Test_RMSE"].max() * 1.20
)

# R²: retain negative values
axes[2].set_xlim(
    min(-0.40, df["Test_R2"].min() - 0.08),
    df["Test_R2"].max() + 0.12
)

# ============================================================
# TITLES
# ============================================================

fig.suptitle(
    "Expanded SWBA Forecasting Model Performance",
    fontsize=16,
    fontweight="bold",
    y=0.98
)

fig.text(
    0.5,
    0.025,
    "Evaluation period: January 2020 – December 2025 "
    "| Mnjoli W60E Catchment",
    ha="center",
    fontsize=10
)

# Reserve space for title and bottom caption
fig.subplots_adjust(
    left=0.14,
    right=0.98,
    top=0.88,
    bottom=0.13
)

# ============================================================
# SAVE
# ============================================================

png_path = OUTPUT_DIR / "expanded_model_performance_comparison.png"
pdf_path = OUTPUT_DIR / "expanded_model_performance_comparison.pdf"

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

print("\nFigures regenerated successfully:")
print(png_path)
print(pdf_path)
