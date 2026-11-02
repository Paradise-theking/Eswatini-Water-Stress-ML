
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# FIGURE 4: INDEPENDENT-TEST PERFORMANCE COMPARISON
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
# 2. OFFICIAL INDEPENDENT-TEST RESULTS
# ------------------------------------------------------------

models = [
    "Final Ridge Regression",
    "Persistence baseline",
    "Climatology baseline (SWBA = 0)"
]

rmse = [
    1.3906,
    1.7568,
    1.7589
]

improvement_persistence = 20.84
improvement_climatology = 20.93

y_positions = np.arange(len(models))


# ------------------------------------------------------------
# 3. CREATE FIGURE
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
# 4. ANNOTATE RMSE VALUES
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
# 5. HIGHLIGHT RIDGE IMPROVEMENT
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
# 6. FORMAT AXES
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

ax.set_xlim(0, 2.15)

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
# 7. FOOTNOTE
# ------------------------------------------------------------

fig.text(
    0.5,
    0.025,
    (
        "Evaluation period: January 2024–December 2025 "
        "| Independent test observations: n = 24"
    ),
    ha="center",
    fontsize=9,
    color="#64748B"
)

fig.tight_layout(rect=[0, 0.07, 1, 1])


# ------------------------------------------------------------
# 8. SAVE PUBLICATION-QUALITY OUTPUTS
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
