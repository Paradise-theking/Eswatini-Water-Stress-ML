from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score
)

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "outputs/expanded/drought_onset_model_preparation/"
    "onset_classification_fold_dataset.csv"
)

OUTPUT_DIR = Path(
    "outputs/expanded/drought_onset_model_preparation/"
    "logistic_regression"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "precipitation_mm_anom",
    "pet_mm_anom",
    "volumetric_soil_water_layer_1_anom",
    "water_balance_3month_anom"
]

TARGET = "onset_label"
THRESHOLD = 0.50

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = (
    FEATURES
    + [
        TARGET,
        "lead_months",
        "split",
        "test_period",
        "forecast_origin_text"
    ]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df[required_columns].isna().any().any():
    raise ValueError(
        "Missing values detected. Resolve before modelling."
    )

df[TARGET] = df[TARGET].astype(int)

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

# ============================================================
# MODEL
# ============================================================

def build_model():

    return Pipeline([
        ("scaler", StandardScaler()),

        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42
        ))
    ])


# ============================================================
# EXPANDING-WINDOW EVALUATION
# ============================================================

fold_results = []
prediction_results = []

leads = sorted(df["lead_months"].unique())
folds = sorted(df["test_period"].unique())

print("=" * 75)
print("DROUGHT-ONSET LOGISTIC REGRESSION")
print("=" * 75)

print("\nFeatures:")
for feature in FEATURES:
    print(" ", feature)

print("\nClassification threshold:", THRESHOLD)

for lead in leads:

    print("\n" + "=" * 75)
    print(f"LEAD TIME: {lead} MONTH(S)")
    print("=" * 75)

    for fold in folds:

        subset = df[
            (df["lead_months"] == lead)
            & (df["test_period"] == fold)
        ].copy()

        train = subset[subset["split"] == "train"].copy()
        test = subset[subset["split"] == "test"].copy()

        if train.empty or test.empty:
            raise ValueError(
                f"Empty training or test set: "
                f"lead={lead}, fold={fold}"
            )

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]

        if y_train.nunique() < 2:
            raise ValueError(
                f"Training set has only one class: "
                f"lead={lead}, fold={fold}"
            )

        model = build_model()

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        pr_auc = (
            average_precision_score(y_test, probabilities)
            if y_test.nunique() == 2
            else np.nan
        )

        metrics = {
            "lead_months": lead,
            "test_period": fold,
            "training_samples": len(train),
            "test_samples": len(test),
            "training_onsets": int(y_train.sum()),
            "test_onsets": int(y_test.sum()),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
            "accuracy": accuracy_score(
                y_test, predictions
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            )
        }

        fold_results.append(metrics)

        for i in range(len(test)):

            prediction_results.append({
                "lead_months": lead,
                "test_period": fold,
                "forecast_origin": (
                    test.iloc[i]["forecast_origin_text"]
                ),
                "actual_onset": int(y_test.iloc[i]),
                "predicted_probability": float(probabilities[i]),
                "predicted_onset": int(predictions[i]),
                "threshold": THRESHOLD
            })

        print(f"\nFold: {fold}")
        print(
            f"Training: {len(train)} samples, "
            f"{int(y_train.sum())} onsets"
        )
        print(
            f"Testing: {len(test)} samples, "
            f"{int(y_test.sum())} onsets"
        )
        print(
            f"TP={tp} | FP={fp} | FN={fn} | TN={tn}"
        )
        print(
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"F1: {f1:.3f}"
        )

# ============================================================
# POOLED OUT-OF-SAMPLE EVALUATION
# ============================================================

fold_df = pd.DataFrame(fold_results)
pred_df = pd.DataFrame(prediction_results)

pooled_results = []

print("\n" + "=" * 75)
print("POOLED OUT-OF-SAMPLE RESULTS")
print("=" * 75)

for lead in leads:

    result = pred_df[
        pred_df["lead_months"] == lead
    ].copy()

    y_true = result["actual_onset"]
    y_pred = result["predicted_onset"]
    probabilities = result["predicted_probability"]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    pr_auc = average_precision_score(
        y_true, probabilities
    )

    pooled_results.append({
        "lead_months": lead,
        "test_samples": len(result),
        "observed_onsets": int(y_true.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        )
    })

    print(f"\nLead: {lead} month(s)")
    print(f"Test observations: {len(result)}")
    print(f"Observed onset cases: {int(y_true.sum())}")
    print(f"TP={tp} | FP={fp} | FN={fn} | TN={tn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")

# ============================================================
# SAVE OUTPUTS
# ============================================================

pooled_df = pd.DataFrame(pooled_results)

fold_path = OUTPUT_DIR / "fold_metrics.csv"
pooled_path = OUTPUT_DIR / "pooled_metrics.csv"
pred_path = OUTPUT_DIR / "out_of_sample_predictions.csv"

fold_df.to_csv(fold_path, index=False)
pooled_df.to_csv(pooled_path, index=False)
pred_df.to_csv(pred_path, index=False)

print("\n" + "=" * 75)
print("MODEL EVALUATION COMPLETE")
print("=" * 75)

print("\nSaved:")
print(fold_path.resolve())
print(pooled_path.resolve())
print(pred_path.resolve())

print("\nNo test-set threshold optimisation was performed.")
print("No additional models were trained.")
