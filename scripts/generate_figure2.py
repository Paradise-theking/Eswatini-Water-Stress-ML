
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --------------------------------------------------
# Figure 2: Chronological Data Partitioning
# --------------------------------------------------

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Verified chronological data partitions
partitions = [
    {
        "name": "Training",
        "start": "Jul 2015",
        "end": "Dec 2021",
        "count": 78,
        "purpose": "Model fitting\nFeature ranking",
        "color": "#2E7D32",
    },
    {
        "name": "Validation",
        "start": "Jan 2022",
        "end": "Dec 2023",
        "count": 24,
        "purpose": "Feature selection\nHyperparameter tuning",
        "color": "#E69F00",
    },
    {
        "name": "Independent Test",
        "start": "Jan 2024",
        "end": "Dec 2025",
        "count": 24,
        "purpose": "Final model\nevaluation",
        "color": "#2878B5",
    },
]

TOTAL = sum(p["count"] for p in partitions)

# Figure setup
fig, ax = plt.subplots(figsize=(13, 5.8))

fig.suptitle(
    "Chronological Partitioning of the Dataset",
    fontsize=19,
    fontweight="bold",
    y=0.95,
)

# Draw proportional timeline
x_position = 0

for p in partitions:
    width = p["count"]

    ax.add_patch(
        Rectangle(
            (x_position, 0.45),
            width,
            0.9,
            facecolor=p["color"],
            edgecolor="white",
            linewidth=2,
        )
    )

    center = x_position + width / 2

    ax.text(
        center, 1.08,
        p["name"],
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="white",
    )

    ax.text(
        center, 0.78,
        f'{p["count"]} observations',
        ha="center",
        va="center",
        fontsize=10,
        color="white",
    )

    ax.text(
        center, 0.12,
        f'{p["start"]} – {p["end"]}\n{p["purpose"]}',
        ha="center",
        va="top",
        fontsize=10,
        linespacing=1.5,
    )

    x_position += width

# Timeline formatting
ax.set_xlim(0, TOTAL)
ax.set_ylim(-0.65, 1.75)
ax.set_yticks([])
ax.set_xticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

ax.text(
    TOTAL / 2,
    1.55,
    f"Total dataset: {TOTAL} monthly observations (July 2015 – December 2025)",
    ha="center",
    fontsize=11,
    color="#333333",
)

fig.text(
    0.5, 0.025,
    "Chronological split: 78 training | 24 validation | 24 independent test observations",
    ha="center",
    fontsize=10,
    color="#444444",
)

plt.tight_layout(rect=[0.03, 0.09, 0.97, 0.90])

# Save high-resolution PNG and PDF
png_path = OUTPUT_DIR / "figure2_chronological_partition.png"
pdf_path = OUTPUT_DIR / "figure2_chronological_partition.pdf"

fig.savefig(png_path, dpi=400, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

plt.show()

print("\nFigure 2 generated successfully!")
print(f"PNG: {png_path}")
print(f"PDF: {pdf_path}")
