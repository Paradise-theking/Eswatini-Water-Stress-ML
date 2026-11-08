
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "feature_selection_validation_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results"


# ============================================================
# 2. LOAD ACTUAL VALIDATION RESULTS
# ============================================================

df = pd.read_csv(CSV_PATH)

required_columns = [
    "Features",
    "MAE",
    "RMSE",
    "R2"
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

df = df.sort_values("Features").reset_index(drop=True)

print("\nFeature Selection Validation Results")
print("=" * 65)
print(df.round(4).to_string(index=False))


# ============================================================
# 3. IDENTIFY BEST FEATURE COUNT
# ============================================================

best_row = df.loc[df["RMSE"].idxmin()]

best_features = int(best_row["Features"])

print("\nBest feature count based on validation RMSE:")
print(f"Features: {best_features}")
print(f"MAE:     {best_row['MAE']:.4f}")
print(f"RMSE:    {best_row['RMSE']:.4f}")
print(f"R²:      {best_row['R2']:.4f}")


# ============================================================
# 4. DEFINE INDIVIDUAL FIGURES
# ============================================================

metrics = [
    {
        "column": "MAE",
        "title": "Feature Selection Performance: MAE",
        "ylabel": "Mean Absolute Error (MAE)",
        "direction": "Lower is better",
        "filename": "feature_selection_MAE.png"
    },
    {
        "column": "RMSE",
        "title": "Feature Selection Performance: RMSE",
        "ylabel": "Root Mean Squared Error (RMSE)",
        "direction": "Lower is better",
        "filename": "feature_selection_RMSE.png"
    },
    {
        "column": "R2",
        "title": "Feature Selection Performance: R²",
        "ylabel": "Coefficient of Determination (R²)",
        "direction": "Higher is better",
        "filename": "feature_selection_R2.png"
    }
]


# ============================================================
# 5. GENERATE AND SAVE EACH BAR GRAPH SEPARATELY
# ============================================================

feature_labels = df["Features"].astype(str)

for metric in metrics:

    column = metric["column"]
    values = df[column]

    # Highlight the optimal feature count
    bar_colors = [
        "steelblue"
        if int(feature_count) == best_features
        else "lightsteelblue"
        for feature_count in df["Features"]
    ]

    # Create a large, separate figure for each metric
    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    bars = ax.bar(
        feature_labels,
        values,
        width=0.65,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.7
    )

    # Titles and axis labels
    ax.set_title(
        metric["title"],
        fontsize=15,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel(
        "Number of Selected Features",
        fontsize=12
    )

    ax.set_ylabel(
        metric["ylabel"],
        fontsize=12
    )

    ax.tick_params(
        axis="both",
        labelsize=11
    )

    # Grid formatting
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.4
    )

    ax.set_axisbelow(True)

    # Display actual metric values above each bar
    ax.bar_label(
        bars,
        labels=[
            f"{value:.3f}"
            for value in values
        ],
        padding=5,
        fontsize=10
    )

    # Indicate metric direction
    ax.text(
        0.98,
        0.96,
        metric["direction"],
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        style="italic"
    )

    # Provide sufficient space for labels
    ax.set_ylim(
        top=values.max() * 1.20
    )

    fig.tight_layout()

    # Save each figure separately at 300 DPI
    output_path = (
        OUTPUT_DIR
        / metric["filename"]
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()

    print(
        f"\nSaved figure: {output_path}"
    )

    plt.close(fig)


# ============================================================
# 6. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 65)
print("ALL THREE FIGURES GENERATED SUCCESSFULLY")
print("=" * 65)

print("1. feature_selection_MAE.png")
print("2. feature_selection_RMSE.png")
print("3. feature_selection_R2.png")

print("\nResolution: 300 DPI")
print("Source: Actual validation metrics from CSV")
