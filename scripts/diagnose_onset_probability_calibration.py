from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILE = (
    ROOT
    / "outputs/expanded/drought_onset_model_preparation/"
    / "false_alarm_diagnostics/classified_out_of_sample_predictions.csv"
)

TRAINING_FILE = (
    ROOT
    / "outputs/expanded/drought_onset_model_preparation/"
    / "onset_classification_fold_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs/expanded/drought_onset_model_preparation/"
    / "probability_calibration"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "fixed_full_four"
LEAD_MONTHS = 1
N_BINS = 5


# ============================================================
# LOAD AND VALIDATE OUT-OF-SAMPLE PREDICTIONS
# ============================================================

predictions = pd.read_csv(PREDICTION_FILE)

required_prediction_columns = [
    "test_period",
    "model",
    "actual_onset",
    "predicted_probability",
]

missing = [
    column for column in required_prediction_columns
    if column not in predictions.columns
]

if missing:
    raise ValueError(
        f"Missing prediction columns: {missing}"
    )

predictions = predictions[
    predictions["model"] == MODEL_NAME
].copy()

predictions["actual_onset"] = pd.to_numeric(
    predictions["actual_onset"],
    errors="raise"
).astype(int)

predictions["predicted_probability"] = pd.to_numeric(
    predictions["predicted_probability"],
    errors="raise"
)

if predictions.empty:
    raise ValueError("No predictions found for the selected model.")

if predictions["actual_onset"].isin([0, 1]).all() is False:
    raise ValueError("Actual onset labels must be binary (0 or 1).")

if not predictions["predicted_probability"].between(0, 1).all():
    raise ValueError("Predicted probabilities must be between 0 and 1.")

if predictions.duplicated(
    subset=["test_period", "forecast_origin"]
).any():
    raise ValueError("Duplicate forecast origins found.")

predictions = predictions.sort_values(
    ["test_period", "target_date"]
).reset_index(drop=True)

print("=" * 65)
print("OUT-OF-SAMPLE PREDICTION AUDIT")
print("=" * 65)
print("Model:", MODEL_NAME)
print("Observations:", len(predictions))
print("Observed onsets:", predictions["actual_onset"].sum())
print(
    "Observed onset prevalence:",
    round(predictions["actual_onset"].mean(), 4)
)


# ============================================================
# LOAD TRAINING DATA FOR FOLD-SPECIFIC BASELINE PREVALENCE
# ============================================================

training_data = pd.read_csv(TRAINING_FILE)

required_training_columns = [
    "test_period",
    "split",
    "lead_months",
    "drought_onset",
]

missing = [
    column for column in required_training_columns
    if column not in training_data.columns
]

if missing:
    raise ValueError(
        f"Missing training dataset columns: {missing}"
    )

training_data["lead_months"] = pd.to_numeric(
    training_data["lead_months"],
    errors="raise"
)

training_data["drought_onset"] = pd.to_numeric(
    training_data["drought_onset"],
    errors="raise"
).astype(int)

fold_baselines = {}

for period in predictions["test_period"].unique():

    fold_training = training_data[
        (training_data["test_period"] == period)
        & (training_data["split"] == "train")
        & (training_data["lead_months"] == LEAD_MONTHS)
    ]

    if fold_training.empty:
        raise ValueError(
            f"No training observations found for fold {period}."
        )

    fold_baselines[period] = (
        fold_training["drought_onset"].mean()
    )

    print(
        f"Training prevalence for {period}: "
        f"{fold_baselines[period]:.4f} "
        f"(n={len(fold_training)})"
    )

predictions["baseline_probability"] = (
    predictions["test_period"].map(fold_baselines)
)

if predictions["baseline_probability"].isna().any():
    raise ValueError("Could not assign a training prevalence baseline.")


# ============================================================
# POOLED BRIER SCORE AND BRIER SKILL SCORE
# ============================================================

y_true = predictions["actual_onset"]
y_probability = predictions["predicted_probability"]
y_baseline = predictions["baseline_probability"]

model_brier = brier_score_loss(
    y_true,
    y_probability
)

baseline_brier = np.mean(
    (y_true - y_baseline) ** 2
)

brier_skill_score = (
    1 - model_brier / baseline_brier
    if baseline_brier > 0
    else np.nan
)

pooled_summary = pd.DataFrame([{
    "model": MODEL_NAME,
    "lead_months": LEAD_MONTHS,
    "observations": len(predictions),
    "observed_onsets": int(y_true.sum()),
    "observed_prevalence": y_true.mean(),
    "model_brier_score": model_brier,
    "training_prevalence_baseline_brier": baseline_brier,
    "brier_skill_score": brier_skill_score,
}])

pooled_summary.to_csv(
    OUTPUT_DIR / "pooled_calibration_metrics.csv",
    index=False
)


# ============================================================
# CALIBRATION BY CHRONOLOGICAL TEST PERIOD
# ============================================================

fold_results = []

for period, group in predictions.groupby("test_period", sort=True):

    actual = group["actual_onset"]
    probability = group["predicted_probability"]
    baseline = group["baseline_probability"]

    model_score = brier_score_loss(actual, probability)

    baseline_score = np.mean(
        (actual - baseline) ** 2
    )

    skill = (
        1 - model_score / baseline_score
        if baseline_score > 0
        else np.nan
    )

    fold_results.append({
        "test_period": period,
        "observations": len(group),
        "observed_onsets": int(actual.sum()),
        "observed_prevalence": actual.mean(),
        "training_prevalence": baseline.iloc[0],
        "model_brier_score": model_score,
        "baseline_brier_score": baseline_score,
        "brier_skill_score": skill,
    })

fold_metrics = pd.DataFrame(fold_results)

fold_metrics.to_csv(
    OUTPUT_DIR / "calibration_by_test_period.csv",
    index=False
)


# ============================================================
# RELIABILITY TABLE: FIVE QUANTILE-BASED BINS
# ============================================================

predictions["calibration_bin"] = pd.qcut(
    predictions["predicted_probability"],
    q=N_BINS,
    duplicates="drop"
)

reliability = (
    predictions
    .groupby("calibration_bin", observed=True)
    .agg(
        observations=("actual_onset", "size"),
        mean_predicted_probability=(
            "predicted_probability", "mean"
        ),
        observed_onset_frequency=("actual_onset", "mean"),
        observed_onsets=("actual_onset", "sum"),
        minimum_probability=("predicted_probability", "min"),
        maximum_probability=("predicted_probability", "max"),
    )
    .reset_index()
)

reliability["calibration_bin"] = (
    reliability["calibration_bin"].astype(str)
)

reliability.to_csv(
    OUTPUT_DIR / "reliability_table.csv",
    index=False
)


# ============================================================
# RELIABILITY DIAGRAM
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Perfect calibration"
)

ax.plot(
    reliability["mean_predicted_probability"],
    reliability["observed_onset_frequency"],
    marker="o",
    linewidth=2,
    markersize=8,
    label="Four-predictor model"
)

for _, row in reliability.iterrows():
    ax.annotate(
        f"n={int(row['observations'])}",
        (
            row["mean_predicted_probability"],
            row["observed_onset_frequency"]
        ),
        xytext=(5, 7),
        textcoords="offset points",
        fontsize=8
    )

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed onset frequency")

ax.set_title(
    "Probability Calibration — One-Month Drought Onset Warning"
)

ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "reliability_diagram.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "reliability_diagram.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# PREDICTED PROBABILITY DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    predictions["predicted_probability"],
    bins=np.linspace(0, 1, 11),
    edgecolor="black"
)

ax.set_xlabel("Predicted onset probability")
ax.set_ylabel("Number of forecasts")

ax.set_title(
    "Distribution of Out-of-Sample Drought Onset Probabilities"
)

ax.set_xlim(0, 1)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "probability_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "probability_distribution.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# SAVE PREDICTIONS WITH BASELINE AND CALIBRATION BIN
# ============================================================

predictions.to_csv(
    OUTPUT_DIR / "calibration_predictions.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("POOLED CALIBRATION RESULTS")
print("=" * 65)

print(pooled_summary.to_string(index=False))

print("\n" + "=" * 65)
print("CALIBRATION BY TEST PERIOD")
print("=" * 65)

print(fold_metrics.to_string(index=False))

print("\n" + "=" * 65)
print("RELIABILITY TABLE")
print("=" * 65)

print(reliability.to_string(index=False))

print("\n" + "=" * 65)
print("FILES GENERATED")
print("=" * 65)

for file in sorted(OUTPUT_DIR.iterdir()):
    print(file.name)

print("\nCalibration diagnostic completed successfully.")
