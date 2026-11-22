from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# FIGURE 4: INDEPENDENT-TEST PERFORMANCE COMPARISON
# Results loaded from Notebook 05 canonical exports
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

performance_path = (
    OUTPUT_DIR / "independent_test_performance.csv"
)

forecasts_path = (
    OUTPUT_DIR / "independent_test_forecasts.csv"
)

if not performance_path.exists():
    raise FileNotFoundError(
        f"Missing evaluation export: {performance_path}"
    )

if not forecasts_path.exists():
    raise FileNotFoundError(
        f"Missing forecast export: {forecasts_path}"
    )


# ------------------------------------------------------------
# 2. LOAD CANONICAL EVALUATION RESULTS
# ------------------------------------------------------------

performance = pd.read_csv(performance_path)

forecasts = pd.read_csv(
    forecasts_path,
    parse_dates=["forecast_date"]
)

required_models = [
    "Final Ridge t+1",
    "Persistence",
    "Climatology (WSI=0)"
]

if set(performance["Model"]) != set(required_models):
    raise ValueError(
        "Unexpected model names in performance export."
    )

if performance["Model"].duplicated().any():
    raise ValueError(
        "Duplicate model entries in performance export."
    )

if len(forecasts) != 24:
    raise ValueError(
        f"Expected 24 test forecasts; found {len(forecasts)}."
    )

if forecasts.isna().any().any():
    raise ValueError(
        "Missing values detected in forecast export."
    )

performance = performance.set_index("Model")

rmse = [
    performance.loc[model, "RMSE"]
    for model in required_models
]

ridge_rmse = performance.loc[
    "Final Ridge t+1", "RMSE"
]

persistence_rmse = performance.loc[
    "Persistence", "RMSE"
]

climatology_rmse = performance.loc[
    "Climatology (WSI=0)", "RMSE"
]


# ------------------------------------------------------------
# 3. CALCULATE IMPROVEMENTS FROM EXPORTED METRICS
# ------------------------------------------------------------

improvement_persistence = (
    (persistence_rmse - ridge_rmse)
    / persistence_rmse
    * 100
)

improvement_climatology = (
    (climatology_rmse - ridge_rmse)
    / climatology_rmse
    * 100
)

models = [
    "Final Ridge Regression",
    "Persistence baseline",
    "Climatology baseline (SWBA = 0)"
]

y_positions = np.arange(len(models))


# ------------------------------------------------------------
# 4. CREATE FIGURE
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    y_positions,
    rmse,
    height=0.55,
    color=["#287A67", "#7895B2", "#A7B5C5"],
    edgecolor="white",
    linewidth=1
)


# ------------------------------------------------------------
# 5. ANNOTATE RMSE VALUES
# ------------------------------------------------------------

for bar, value in zip(bars, rmse):

    ax.text(
        value + 0.025,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.4f}",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#263238"
    )


# ------------------------------------------------------------
# 6. HIGHLIGHT RIDGE IMPROVEMENT
# ------------------------------------------------------------

ax.text(
    0.98,
    0.96,
    (
        f"RMSE improvement\n"
        f"vs persistence: {improvement_persistence:.2f}%\n"
        f"vs climatology: {improvement_climatology:.2f}%"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    linespacing=1.5,
    color="#245B4D",
    bbox=dict(
        boxstyle="round,pad=0.55",
        facecolor="#E8F3EF",
        edgecolor="#A8CBBE"
    )
)


# ------------------------------------------------------------
# 7. FORMAT AXES
# ------------------------------------------------------------

ax.set_yticks(y_positions)
ax.set_yticklabels(models, fontsize=10)

ax.invert_yaxis()

ax.set_xlabel(
    "Root Mean Squared Error (RMSE)",
    fontsize=11
)

ax.set_title(
    "Independent-Test Forecast Performance Comparison",
    fontsize=14,
    fontweight="bold",
    pad=18
)

ax.set_xlim(0, max(rmse) * 1.25)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.25
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)


# ------------------------------------------------------------
# 8. DYNAMIC EVALUATION FOOTNOTE
# ------------------------------------------------------------

start_date = forecasts["forecast_date"].min()
end_date = forecasts["forecast_date"].max()
sample_size = len(forecasts)

fig.text(
    0.5,
    0.025,
    (
        f"Evaluation period: {start_date:%B %Y}–"
        f"{end_date:%B %Y} "
        f"| Independent test observations: n = {sample_size}"
    ),
    ha="center",
    fontsize=9,
    color="#64748B"
)

fig.tight_layout(rect=[0, 0.07, 1, 1])


# ------------------------------------------------------------
# 9. SAVE PUBLICATION-QUALITY OUTPUTS
# ------------------------------------------------------------

png_path = OUTPUT_DIR / "figure4_baseline_comparison.png"
pdf_path = OUTPUT_DIR / "figure4_baseline_comparison.pdf"

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

print("\nFigure 4 generated successfully!")
print("PNG:", png_path)
print("PDF:", pdf_path)

print("\nSource: independent_test_performance.csv")
print("Evaluation observations:", sample_size)
print(f"RMSE improvement vs persistence: {improvement_persistence:.2f}%")
print(f"RMSE improvement vs climatology: {improvement_climatology:.2f}%")
