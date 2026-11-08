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
    balanced_accuracy_score,
    roc_auc_score
)

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "outputs/expanded/drought_onset_model_preparation/"
    "origin_valid_sensitivity/"
    "origin_valid_onset_fold_dataset.csv"
)

OUTPUT_DIR = Path(
    "outputs/expanded/drought_onset_model_preparation/"
    "origin_valid_sensitivity/logistic_regression"
)

ORIGINAL_RESULTS = Path(
    "outputs/expanded/drought_onset_model_preparation/"
    "logistic_regression/pooled_metrics.csv"
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
# LOAD AND VALIDATE DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

required = FEATURES + [
    TARGET,
    "lead_months",
    "split",
    "test_period",
    "forecast_origin_text",
    "historical_reference_n"
]

missing = [
    column for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

if df[required].isna().any().any():
    raise ValueError(
        "Missing values found in modelling columns."
    )

df[TARGET] = df[TARGET].astype(int)

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

if (df["historical_reference_n"] < 15).any():
    raise ValueError(
        "Some observations have fewer than 15 historical records."
    )

if (df["historical_reference_n"] < 0).any():
    raise ValueError("Invalid historical reference count.")

# ============================================================
# MODEL — IDENTICAL TO ORIGINAL EXPERIMENT
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
print("ORIGIN-VALID DROUGHT-ONSET SENSITIVITY MODEL")
print("=" * 75)

print("\nModel: Logistic Regression")
print("Scaling: StandardScaler")
print("Class weighting: balanced")
print("Regularisation C: 1.0")
print("Solver: liblinear")
print("Random state: 42")
print("Threshold:", THRESHOLD)
print("Minimum historical reference:", 15)

print("\nPredictors:")

for feature in FEATURES:
    print(" ", feature)

for lead in leads:

    print("\n" + "=" * 75)
    print(f"FORECAST LEAD: {lead} MONTH(S)")
    print("=" * 75)

    for fold in folds:

        subset = df[
            (df["lead_months"] == lead)
            & (df["test_period"] == fold)
        ].copy()

        train = subset[
            subset["split"] == "train"
        ].copy()

        test = subset[
            subset["split"] == "test"
        ].copy()

        if train.empty or test.empty:
            raise ValueError(
                f"Empty dataset: lead={lead}, fold={fold}"
            )

        if train[TARGET].nunique() < 2:
            raise ValueError(
                f"Training set has one class: "
                f"lead={lead}, fold={fold}"
            )

        if test[TARGET].nunique() < 2:
            raise ValueError(
                f"Test set has one class: "
                f"lead={lead}, fold={fold}"
            )

        # Check that each fold has unique target periods.
        if train["target_period_text"].duplicated().any():
            raise ValueError(
                f"Duplicate training target periods: {fold}"
            )

        if test["target_period_text"].duplicated().any():
            raise ValueError(
                f"Duplicate test target periods: {fold}"
            )

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]

        model = build_model()

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

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

        pr_auc = average_precision_score(
            y_test,
            probabilities
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
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
            "roc_auc": roc_auc,
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
                "target_period": (
                    test.iloc[i]["target_period_text"]
                ),
                "forecast_origin": (
                    test.iloc[i]["forecast_origin_text"]
                ),
                "actual_onset": int(y_test.iloc[i]),
                "predicted_probability": float(
                    probabilities[i]
                ),
                "predicted_onset": int(predictions[i]),
                "threshold": THRESHOLD,
                "historical_reference_n": int(
                    test.iloc[i]["historical_reference_n"]
                )
            })

        print(f"\nFold: {fold}")

        print(
            f"Training: {len(train)} observations, "
            f"{int(y_train.sum())} onsets"
        )

        print(
            f"Testing: {len(test)} observations, "
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

        print(
            f"PR-AUC: {pr_auc:.3f} | "
            f"ROC-AUC: {roc_auc:.3f}"
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

    pooled_results.append({
        "lead_months": lead,
        "test_samples": len(result),
        "observed_onsets": int(y_true.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
        "accuracy": accuracy_score(
            y_true, y_pred
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        )
    })

pooled_df = pd.DataFrame(pooled_results)

for _, row in pooled_df.iterrows():

    print(f"\nLead: {int(row['lead_months'])} month(s)")

    print(
        f"Observations: {int(row['test_samples'])} | "
        f"Onsets: {int(row['observed_onsets'])}"
    )

    print(
        f"TP={int(row['true_positives'])} | "
        f"FP={int(row['false_positives'])} | "
        f"FN={int(row['false_negatives'])} | "
        f"TN={int(row['true_negatives'])}"
    )

    print(
        f"Precision: {row['precision']:.3f} | "
        f"Recall: {row['recall']:.3f} | "
        f"F1: {row['f1']:.3f}"
    )

    print(
        f"PR-AUC: {row['pr_auc']:.3f} | "
        f"ROC-AUC: {row['roc_auc']:.3f}"
    )

# ============================================================
# COMPARE AGAINST ORIGINAL MODEL
# ============================================================

comparison_path = OUTPUT_DIR / "original_vs_origin_valid.csv"

if ORIGINAL_RESULTS.exists():

    original = pd.read_csv(ORIGINAL_RESULTS)

    common_metrics = [
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "accuracy",
        "balanced_accuracy"
    ]

    original_columns = [
        "lead_months"
    ] + [
        metric for metric in common_metrics
        if metric in original.columns
    ]

    sensitivity_columns = [
        "lead_months"
    ] + common_metrics

    comparison = pooled_df[
        sensitivity_columns
    ].merge(
        original[original_columns],
        on="lead_months",
        how="inner",
        suffixes=("_origin_valid", "_original"),
        validate="one_to_one"
    )

    for metric in common_metrics:

        new_col = f"{metric}_origin_valid"
        old_col = f"{metric}_original"

        if new_col in comparison.columns and old_col in comparison.columns:

            comparison[f"{metric}_change"] = (
                comparison[new_col] - comparison[old_col]
            )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print("\n" + "=" * 75)
    print("ORIGINAL VS ORIGIN-VALID COMPARISON")
    print("=" * 75)

    print(comparison.to_string(index=False))

else:

    print("\nOriginal pooled results not found.")
    print("Comparison will be completed separately.")

# ============================================================
# SAVE ALL RESULTS
# ============================================================

fold_df.to_csv(
    OUTPUT_DIR / "origin_valid_fold_metrics.csv",
    index=False
)

pooled_df.to_csv(
    OUTPUT_DIR / "origin_valid_pooled_metrics.csv",
    index=False
)

pred_df.to_csv(
    OUTPUT_DIR / "origin_valid_out_of_sample_predictions.csv",
    index=False
)

print("\n" + "=" * 75)
print("ORIGIN-VALID MODEL EVALUATION COMPLETE")
print("=" * 75)

print("\nSaved outputs:")

for file in OUTPUT_DIR.glob("*.csv"):
    print(file.resolve())

print("\nOriginal model files were not modified.")
print("No test-set threshold optimisation was performed.")
