from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "expanded"
    / "final_evaluation"
    / "final_test_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "expanded"
    / "final_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(RESULTS_PATH)

df = df.sort_values(
    "Test_RMSE",
    ascending=True
).reset_index(drop=True)

models = [
    "Lasso",
    "Ridge",
    "XGBoost",
    "Persistence",
    "Seasonal Climatology",
    "Training-Mean Baseline",
]

df["Display_Model"] = df["Model"].replace({
    "Tuned XGBoost": "XGBoost"
})

# ============================================================
# PUBLICATION STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ============================================================
# FIGURE GENERATOR
# ============================================================

def generate_metric_figure(
    metric_column,
    title,
    xlabel,
    filename,
    higher_is_better=False
):

    plot_df = df.copy()

    values = plot_df[metric_column].values
    labels = plot_df["Display_Model"].values

    fig, ax = plt.subplots(figsize=(10, 6.5))

    y_positions = np.arange(len(labels))

    bars = ax.barh(
        y_positions,
        values,
        height=0.62
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)

    ax.invert_yaxis()

    ax.set_title(
        title,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel(xlabel, labelpad=10)

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # VALUE LABELS
    # --------------------------------------------------------

    for bar, value in zip(bars, values):

        y = bar.get_y() + bar.get_height() / 2

        if value >= 0:

            ax.annotate(
                f"{value:.3f}",
                xy=(value, y),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold"
            )

        else:

            ax.annotate(
                f"{value:.3f}",
                xy=(value, y),
                xytext=(-7, 0),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=10,
                fontweight="bold"
            )

    # --------------------------------------------------------
    # AXIS LIMITS
    # --------------------------------------------------------

    if metric_column in ["Test_MAE", "Test_RMSE"]:

        ax.set_xlim(
            0,
            max(values) * 1.20
        )

        direction = "Lower values indicate better performance"

    else:

        ax.axvline(
            0,
            linewidth=1,
            linestyle="-",
            alpha=0.7
        )

        ax.set_xlim(
            min(-0.40, min(values) - 0.10),
            max(values) + 0.15
        )

        direction = "Higher values indicate better performance"

    # --------------------------------------------------------
    # SUBTITLE / CONTEXT
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.025,
        "Mnjoli W60E Catchment | January 2020 – December 2025",
        ha="center",
        fontsize=10
    )

    fig.text(
        0.5,
        0.065,
        direction,
        ha="center",
        fontsize=9,
        style="italic"
    )

    fig.subplots_adjust(
        left=0.25,
        right=0.95,
        top=0.87,
        bottom=0.17
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    png_path = OUTPUT_DIR / f"{filename}.png"
    pdf_path = OUTPUT_DIR / f"{filename}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nGenerated: {png_path.name}")
    print(f"Generated: {pdf_path.name}")


# ============================================================
# GENERATE THREE SEPARATE FIGURES
# ============================================================

generate_metric_figure(
    metric_column="Test_MAE",
    title="Model Comparison: Mean Absolute Error",
    xlabel="Mean Absolute Error (MAE)",
    filename="expanded_model_comparison_MAE"
)

generate_metric_figure(
    metric_column="Test_RMSE",
    title="Model Comparison: Root Mean Squared Error",
    xlabel="Root Mean Squared Error (RMSE)",
    filename="expanded_model_comparison_RMSE"
)

generate_metric_figure(
    metric_column="Test_R2",
    title="Model Comparison: Coefficient of Determination",
    xlabel="Coefficient of Determination (R²)",
    filename="expanded_model_comparison_R2",
    higher_is_better=True
)

print("\n" + "=" * 65)
print("ALL THREE FIGURES GENERATED SUCCESSFULLY")
print("=" * 65)
