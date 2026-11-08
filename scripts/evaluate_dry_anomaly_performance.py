from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "expanded"
    / "final_evaluation"
    / "final_test_predictions.csv"
)

OUTPUT_DIR = INPUT_PATH.parent

DRY_THRESHOLD = -1.0

MODEL_COLUMNS = [
    "Lasso",
    "Ridge",
    "Tuned XGBoost",
    "Persistence",
    "Seasonal Climatology",
    "Training-Mean Baseline",
]

# ============================================================
# LOAD PREDICTIONS
# ============================================================

df = pd.read_csv(INPUT_PATH)

df["target_month"] = pd.to_datetime(df["target_month"])

required_columns = ["observed_swba"] + MODEL_COLUMNS

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df = df.dropna(subset=["observed_swba"]).copy()

# ============================================================
# IDENTIFY OBSERVED DRY-ANOMALY MONTHS
# ============================================================

df["observed_dry"] = (
    df["observed_swba"] <= DRY_THRESHOLD
)

dry_df = df[df["observed_dry"]].copy()

n_total = len(df)
n_dry = len(dry_df)
n_normal = n_total - n_dry

print("=" * 75)
print("DRY-ANOMALY FORECAST EVALUATION")
print("=" * 75)

print(f"\nEvaluation observations: {n_total}")
print(f"Observed dry-anomaly months (SWBA <= -1): {n_dry}")
print(f"Other months: {n_normal}")

if n_dry > 0:
    print(
        "\nObserved dry-anomaly period:",
        dry_df["target_month"].min().strftime("%Y-%m"),
        "to",
        dry_df["target_month"].max().strftime("%Y-%m"),
    )
else:
    print("\nNo observed dry-anomaly months were found.")

# ============================================================
# EVALUATE EACH MODEL
# ============================================================

results = []

for model_name in MODEL_COLUMNS:

    observed = df["observed_swba"].to_numpy(dtype=float)
    predicted = df[model_name].to_numpy(dtype=float)

    # Dry-anomaly warning classification
    actual_dry = observed <= DRY_THRESHOLD
    predicted_dry = predicted <= DRY_THRESHOLD

    tp = int(np.sum(actual_dry & predicted_dry))
    fp = int(np.sum(~actual_dry & predicted_dry))
    fn = int(np.sum(actual_dry & ~predicted_dry))
    tn = int(np.sum(~actual_dry & ~predicted_dry))

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    if pd.notna(precision) and pd.notna(recall):
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
    else:
        f1 = np.nan

    accuracy = (tp + tn) / n_total if n_total else np.nan

    # Continuous forecast errors during observed dry months
    if n_dry > 0:

        dry_observed = dry_df["observed_swba"].to_numpy(dtype=float)
        dry_predicted = dry_df[model_name].to_numpy(dtype=float)

        dry_mae = np.mean(
            np.abs(dry_observed - dry_predicted)
        )

        dry_rmse = np.sqrt(
            np.mean((dry_observed - dry_predicted) ** 2)
        )

        dry_bias = np.mean(
            dry_predicted - dry_observed
        )

    else:
        dry_mae = np.nan
        dry_rmse = np.nan
        dry_bias = np.nan

    results.append({
        "Model": model_name,
        "Dry_Months": n_dry,
        "Dry_Month_MAE": dry_mae,
        "Dry_Month_RMSE": dry_rmse,
        "Dry_Month_Bias": dry_bias,
        "True_Positive": tp,
        "False_Positive": fp,
        "False_Negative": fn,
        "True_Negative": tn,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1_Score": f1,
        "Alert_Accuracy": accuracy,
    })

results_df = pd.DataFrame(results)

# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    OUTPUT_DIR / "dry_anomaly_model_evaluation.csv"
)

results_df.to_csv(output_path, index=False)

# Also save the observed dry-anomaly months for traceability
dry_months_path = (
    OUTPUT_DIR / "observed_dry_anomaly_test_months.csv"
)

dry_df[
    ["target_month", "observed_swba"]
].to_csv(dry_months_path, index=False)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)
print("PERFORMANCE DURING OBSERVED DRY-ANOMALY MONTHS")
print("=" * 75)

print(
    results_df[
        [
            "Model",
            "Dry_Months",
            "Dry_Month_MAE",
            "Dry_Month_RMSE",
            "Dry_Month_Bias",
        ]
    ].to_string(
        index=False,
        formatters={
            "Dry_Month_MAE": lambda x: f"{x:.4f}",
            "Dry_Month_RMSE": lambda x: f"{x:.4f}",
            "Dry_Month_Bias": lambda x: f"{x:.4f}",
        }
    )
)

print("\n" + "=" * 75)
print("DRY-ANOMALY WARNING DETECTION")
print("=" * 75)

print(
    results_df[
        [
            "Model",
            "True_Positive",
            "False_Positive",
            "False_Negative",
            "Precision",
            "Recall",
            "F1_Score",
        ]
    ].to_string(
        index=False,
        formatters={
            "Precision": lambda x: f"{x:.3f}",
            "Recall": lambda x: f"{x:.3f}",
            "F1_Score": lambda x: f"{x:.3f}",
        }
    )
)

print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)

print(output_path)
print(dry_months_path)

print("\nDry-anomaly evaluation completed.")
