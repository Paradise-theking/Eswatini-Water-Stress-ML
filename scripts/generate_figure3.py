
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# FIGURE 3: MODEL DEVELOPMENT AND OPTIMIZATION WORKFLOW
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
    raise FileNotFoundError(
        "Could not locate the project root."
    )

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. FIGURE INITIALIZATION
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 10))

ax.set_xlim(0, 13)
ax.set_ylim(0, 12)
ax.axis("off")

fig.patch.set_facecolor("white")
ax.set_facecolor("white")


# ------------------------------------------------------------
# 3. DRAWING FUNCTIONS
# ------------------------------------------------------------

def add_box(x, y, width, height, title, description,
            facecolor="#F3F6FA",
            edgecolor="#34495E",
            title_color="#1F2937"):

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor
    )

    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=title_color
    )

    ax.text(
        x + width / 2,
        y + height * 0.34,
        description,
        ha="center",
        va="center",
        fontsize=9,
        color="#374151",
        linespacing=1.4
    )


def add_arrow(x1, y1, x2, y2):

    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.5,
        color="#64748B"
    )

    ax.add_patch(arrow)


# ------------------------------------------------------------
# 4. WORKFLOW
# ------------------------------------------------------------

# Dataset
add_box(
    3.3, 10.45, 6.4, 1.0,
    "LEAKAGE-FREE FORECASTING DATASET",
    "126 monthly observations | 59 candidate predictors",
    "#E8F0FE",
    "#356AC3"
)

# Training-only ranking
add_arrow(6.5, 10.45, 6.5, 10.05)

add_box(
    3.3, 8.85, 6.4, 1.15,
    "TRAINING-ONLY FEATURE RANKING",
    "78 training observations | Absolute Pearson correlation",
    "#EAF3FF",
    "#356AC3"
)

# Feature selection
add_arrow(6.5, 8.85, 6.5, 8.45)

add_box(
    3.3, 7.15, 6.4, 1.3,
    "VALIDATION-BASED FEATURE SELECTION",
    "7 candidate feature-set sizes: 5, 10, 15, 20, 30, 40, 50\n24 validation observations | Ridge α = 1.0",
    "#FFF7E6",
    "#D69E2E"
)

# Selected feature count
add_arrow(6.5, 7.15, 6.5, 6.75)

add_box(
    3.3, 5.55, 6.4, 1.2,
    "OPTIMAL FEATURE SET",
    "15 predictors selected | Validation RMSE = 1.3234",
    "#FFF7E6",
    "#D69E2E"
)

# Hyperparameter tuning
add_arrow(6.5, 5.55, 6.5, 5.15)

add_box(
    3.3, 3.85, 6.4, 1.3,
    "RIDGE HYPERPARAMETER OPTIMIZATION",
    "11 candidate α values evaluated\n15 predictors held fixed | Validation-only tuning",
    "#F0EBFA",
    "#805AD5"
)

# Final model
add_arrow(6.5, 3.85, 6.5, 3.45)

add_box(
    3.3, 2.25, 6.4, 1.2,
    "FROZEN FINAL MODEL SPECIFICATION",
    "Ridge Regression | 15 predictors | α = 0.05\nValidation RMSE = 1.3150",
    "#E8F5EE",
    "#27845B"
)

# Independent test
add_arrow(6.5, 2.25, 6.5, 1.85)

add_box(
    3.3, 0.65, 6.4, 1.2,
    "INDEPENDENT MODEL EVALUATION",
    "24 unseen observations | January 2024 – December 2025\nNo test data used during model selection or tuning",
    "#FDECEC",
    "#C53030"
)


# ------------------------------------------------------------
# 5. VALIDATION AND TEST PERIOD ANNOTATIONS
# ------------------------------------------------------------

ax.text(
    10.0, 7.75,
    "VALIDATION\n2022–2023",
    ha="left",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="#B7791F"
)

ax.text(
    10.0, 1.25,
    "TEST SET\n2024–2025",
    ha="left",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="#C53030"
)

# Training annotation
ax.text(
    1.0, 9.4,
    "TRAINING\n2015–2021\n78 observations",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="#356AC3"
)


# ------------------------------------------------------------
# 6. TITLE AND FOOTNOTE
# ------------------------------------------------------------

ax.text(
    6.5, 11.75,
    "Model Development and Optimization Workflow",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="#172B4D"
)

ax.text(
    6.5, 0.18,
    "Feature ranking and model fitting used training data only; "
    "validation data guided model selection.",
    ha="center",
    va="center",
    fontsize=8,
    color="#64748B"
)


# ------------------------------------------------------------
# 7. SAVE OUTPUTS
# ------------------------------------------------------------

fig.tight_layout()

png_path = OUTPUT_DIR / "figure3_model_development.png"
pdf_path = OUTPUT_DIR / "figure3_model_development.pdf"

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

print("\nFigure 3 generated successfully!")
print("PNG:", png_path)
print("PDF:", pdf_path)
