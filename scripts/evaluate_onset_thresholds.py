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
    average_precision_score
)

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/expanded/"
    "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)

OUTPUT_DIR = Path(
    "outputs/expanded/drought_onset_model_preparation/"
    "balanced_threshold_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "precipitation_mm",
    "pet_mm",
    "volumetric_soil_water_layer_1",
    "water_balance_3month"
]

TEST_PERIODS = [
    (2006, 2010),
    (2011, 2015),
    (2016, 2020),
    (2021, 2025)
]

# ============================================================
# LOAD AND PREPARE HISTORICAL DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["month_date"] = pd.to_datetime(df["month_date"])

df = df.sort_values("month_date").reset_index(drop=True)

required = [
    "month_date",
    "swba_3month"
] + FEATURES

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing source columns: {missing}")

df["previous_swba"] = df["swba_3month"].shift(1)

df["onset_label"] = (
    (df["swba_3month"] <= -1)
    & (df["previous_swba"] > -1)
).astype(int)

# The first row cannot have a valid previous SWBA.
df.loc[df["previous_swba"].isna(), "onset_label"] = np.nan

# Build forecast-origin records.
origin_data = df[
    ["month_date", "swba_3month"] + FEATURES
].copy()

origin_data = origin_data.rename(
    columns={
        "month_date": "forecast_origin",
        "swba_3month": "origin_swba"
    }
)

print("=" * 75)
print("CHRONOLOGICAL DROUGHT-ONSET THRESHOLD VALIDATION")
print("=" * 75)

# ============================================================
# BUILD LEAD-SPECIFIC FORECAST SAMPLES
# ============================================================

def build_samples(lead):

    targets = df[
        ["month_date", "swba_3month", "onset_label"]
    ].copy()

    targets = targets.dropna(
        subset=["swba_3month", "onset_label"]
    )

    targets["forecast_origin"] = (
        targets["month_date"]
        - pd.DateOffset(months=lead)
    )

    samples = targets.merge(
        origin_data,
        on="forecast_origin",
        how="inner"
    )

    # Genuine advance-warning opportunity:
    # the origin month must not already be in drought.
    samples = samples[
        samples["origin_swba"] > -1
    ].copy()

    samples["onset_label"] = (
        samples["onset_label"].astype(int)
    )

    samples["origin_month"] = (
        samples["forecast_origin"].dt.month
    )

    return samples.reset_index(drop=True)


# ============================================================
# FOLD-SPECIFIC ANOMALY TRANSFORMATION
# ============================================================

def transform_features(samples, cutoff):

    reference = df[
        df["month_date"] <= cutoff
    ].copy()

    if reference.empty:
        raise ValueError(
            f"No reference data available by {cutoff}"
        )

    reference["calendar_month"] = (
        reference["month_date"].dt.month
    )

    output = samples.copy()

    for feature in FEATURES:

        stats = reference.groupby(
            "calendar_month"
        )[feature].agg(
            mean="mean",
            std=lambda x: x.std(ddof=0)
        )

        months = output["origin_month"]

        means = months.map(stats["mean"])
        stds = months.map(stats["std"])

        if means.isna().any() or stds.isna().any():
            raise ValueError(
                f"Missing climatology for {feature}"
            )

        if (stds <= 0).any():
            raise ValueError(
                f"Non-positive standard deviation for {feature}"
            )

        output[f"{feature}_anom"] = (
            output[feature].to_numpy() - means.to_numpy()
        ) / stds.to_numpy()

    return output


ANOMALY_FEATURES = [
    f"{feature}_anom" for feature in FEATURES
]


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
# SELECT THRESHOLD USING VALIDATION DATA ONLY
# ============================================================

def select_threshold(y_true, probabilities):

    candidates = np.unique(
        np.concatenate([
            np.array([0.0]),
            probabilities,
            np.array([1.000001])
        ])
    )

    candidates_metrics = []

    for threshold in candidates:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        candidates_metrics.append({
            "threshold": float(threshold),
            "precision": precision_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "recall": recall_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "f1": f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        })

    # Maximize validation F1; use precision then threshold to break ties.
    selected = max(
        candidates_metrics,
        key=lambda x: (
            x["f1"],
            x["precision"],
            x["threshold"]
        )
    )

    return selected


# ============================================================
# EVALUATION HELPER
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "samples": len(y_true),
        "observed_onsets": int(np.sum(y_true)),
        "predicted_onsets": int(np.sum(predictions)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        )
    }


# ============================================================
# RUN NESTED CHRONOLOGICAL VALIDATION
# ============================================================

all_validation_results = []
all_test_results = []
all_predictions = []

for lead in [1, 2, 3]:

    samples = build_samples(lead)

    print("\n" + "=" * 75)
    print(f"LEAD TIME: {lead} MONTH(S)")
    print("=" * 75)

    for test_start, test_end in TEST_PERIODS:

        validation_start = test_start - 5
        validation_end = test_start - 1

        validation_first_target = pd.Timestamp(
            year=validation_start,
            month=1,
            day=1
        )

        test_first_target = pd.Timestamp(
            year=test_start,
            month=1,
            day=1
        )

        # At the first forecast origin, only labels whose
        # target month is already observed may enter training.
        validation_cutoff = (
            validation_first_target
            - pd.DateOffset(months=lead)
        )

        test_cutoff = (
            test_first_target
            - pd.DateOffset(months=lead)
        )

        validation_train = samples[
            samples["month_date"] <= validation_cutoff
        ].copy()

        validation = samples[
            (samples["month_date"].dt.year >= validation_start)
            & (samples["month_date"].dt.year <= validation_end)
        ].copy()

        test_train = samples[
            samples["month_date"] <= test_cutoff
        ].copy()

        test = samples[
            (samples["month_date"].dt.year >= test_start)
            & (samples["month_date"].dt.year <= test_end)
        ].copy()

        if validation.empty or test.empty:
            raise ValueError(
                f"Empty validation/test data for lead {lead}, "
                f"test period {test_start}-{test_end}"
            )

        if validation_train["onset_label"].nunique() < 2:
            raise ValueError(
                f"Validation training data has only one class: "
                f"lead {lead}, {test_start}-{test_end}"
            )

        if test_train["onset_label"].nunique() < 2:
            raise ValueError(
                f"Test training data has only one class: "
                f"lead {lead}, {test_start}-{test_end}"
            )

        # ----------------------------------------------------
        # STAGE 1: VALIDATION MODEL
        # ----------------------------------------------------

        val_train = transform_features(
            validation_train,
            validation_cutoff
        )

        val_data = transform_features(
            validation,
            validation_cutoff
        )

        val_model = build_model()

        val_model.fit(
            val_train[ANOMALY_FEATURES],
            val_train["onset_label"]
        )

        val_probabilities = val_model.predict_proba(
            val_data[ANOMALY_FEATURES]
        )[:, 1]

        selected = select_threshold(
            val_data["onset_label"].to_numpy(),
            val_probabilities
        )

        threshold = selected["threshold"]

        val_metrics = calculate_metrics(
            val_data["onset_label"].to_numpy(),
            val_probabilities,
            threshold
        )

        val_record = {
            "lead_months": lead,
            "test_period": f"{test_start}-{test_end}",
            "validation_period": (
                f"{validation_start}-{validation_end}"
            ),
            "validation_cutoff": str(
                validation_cutoff.date()
            ),
            "validation_training_samples": len(val_train),
            "validation_samples": len(val_data),
            "validation_onsets": int(
                val_data["onset_label"].sum()
            ),
            "selected_threshold": threshold,
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_pr_auc": val_metrics["pr_auc"]
        }

        all_validation_results.append(val_record)

        # ----------------------------------------------------
        # STAGE 2: REFIT USING ALL PRE-TEST DATA
        # ----------------------------------------------------

        final_train = transform_features(
            test_train,
            test_cutoff
        )

        test_data = transform_features(
            test,
            test_cutoff
        )

        final_model = build_model()

        final_model.fit(
            final_train[ANOMALY_FEATURES],
            final_train["onset_label"]
        )

        test_probabilities = final_model.predict_proba(
            test_data[ANOMALY_FEATURES]
        )[:, 1]

        test_metrics = calculate_metrics(
            test_data["onset_label"].to_numpy(),
            test_probabilities,
            threshold
        )

        test_record = {
            "lead_months": lead,
            "test_period": f"{test_start}-{test_end}",
            "validation_period": (
                f"{validation_start}-{validation_end}"
            ),
            "training_samples": len(final_train),
            "test_cutoff": str(test_cutoff.date()),
            **test_metrics
        }

        all_test_results.append(test_record)

        # ----------------------------------------------------
        # SAVE INDIVIDUAL TEST PREDICTIONS
        # ----------------------------------------------------

        for i in range(len(test_data)):

            all_predictions.append({
                "lead_months": lead,
                "test_period": f"{test_start}-{test_end}",
                "forecast_origin": (
                    test_data.iloc[i]["forecast_origin"]
                    .strftime("%Y-%m")
                ),
                "target_month": (
                    test_data.iloc[i]["month_date"]
                    .strftime("%Y-%m")
                ),
                "actual_onset": int(
                    test_data.iloc[i]["onset_label"]
                ),
                "predicted_probability": float(
                    test_probabilities[i]
                ),
                "selected_threshold": threshold,
                "predicted_onset": int(
                    test_probabilities[i] >= threshold
                )
            })

        print(
            f"\nTest period: {test_start}-{test_end}"
        )
        print(
            f"Validation period: "
            f"{validation_start}-{validation_end}"
        )
        print(
            f"Validation onset cases: "
            f"{int(val_data['onset_label'].sum())}"
        )
        print(
            f"Selected threshold: {threshold:.4f}"
        )
        print(
            f"Validation precision: "
            f"{val_metrics['precision']:.3f}"
        )
        print(
            f"Validation recall: "
            f"{val_metrics['recall']:.3f}"
        )

        print("External test results:")
        print(
            f"TP={test_metrics['true_positives']} | "
            f"FP={test_metrics['false_positives']} | "
            f"FN={test_metrics['false_negatives']} | "
            f"TN={test_metrics['true_negatives']}"
        )
        print(
            f"Precision={test_metrics['precision']:.3f} | "
            f"Recall={test_metrics['recall']:.3f} | "
            f"F1={test_metrics['f1']:.3f} | "
            f"PR-AUC={test_metrics['pr_auc']:.3f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

validation_df = pd.DataFrame(all_validation_results)
test_df = pd.DataFrame(all_test_results)
predictions_df = pd.DataFrame(all_predictions)

validation_df.to_csv(
    OUTPUT_DIR / "validation_threshold_selection.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "external_test_fold_metrics.csv",
    index=False
)

predictions_df.to_csv(
    OUTPUT_DIR / "external_test_predictions.csv",
    index=False
)

# Pooled results across all four chronological test periods.
pooled = []

for lead in [1, 2, 3]:

    subset = predictions_df[
        predictions_df["lead_months"] == lead
    ]

    y_true = subset["actual_onset"].to_numpy()
    probabilities = subset[
        "predicted_probability"
    ].to_numpy()

    predictions = subset[
        "predicted_onset"
    ].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    pooled.append({
        "lead_months": lead,
        "samples": len(subset),
        "observed_onsets": int(y_true.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        )
    })

pooled_df = pd.DataFrame(pooled)

pooled_df.to_csv(
    OUTPUT_DIR / "pooled_threshold_validation_results.csv",
    index=False
)

print("\n" + "=" * 75)
print("POOLED EXTERNAL TEST RESULTS")
print("=" * 75)

print(pooled_df.round(4).to_string(index=False))

print("\n" + "=" * 75)
print("THRESHOLD VALIDATION COMPLETE")
print("=" * 75)

print("\nSaved files:")
for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print(file.resolve())

print(
    "\nThresholds were selected using historical validation "
    "data, not external test outcomes."
)
