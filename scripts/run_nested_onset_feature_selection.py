from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed" / "expanded" /
    "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)

OUTPUT = (
    ROOT / "outputs" / "expanded" /
    "drought_onset_model_preparation" /
    "nested_feature_selection"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

DATE_COL = "month_date"
SWBA_COL = "swba_3month"
DROUGHT_THRESHOLD = -1.0
CLASSIFICATION_THRESHOLD = 0.5

FEATURE_SETS = {
    "rainfall": [
        "precipitation_mm"
    ],
    "PET": [
        "pet_mm"
    ],
    "soil_moisture": [
        "volumetric_soil_water_layer_1"
    ],
    "water_balance": [
        "water_balance_3month"
    ],
    "rainfall_PET_soil": [
        "precipitation_mm",
        "pet_mm",
        "volumetric_soil_water_layer_1"
    ],
    "rainfall_PET_balance": [
        "precipitation_mm",
        "pet_mm",
        "water_balance_3month"
    ],
    "full_four_predictors": [
        "precipitation_mm",
        "pet_mm",
        "volumetric_soil_water_layer_1",
        "water_balance_3month"
    ],
}

FOLDS = [
    {
        "test": "2006-2010",
        "validation": "2001-2005",
        "test_start": "2006-01-01",
        "test_end": "2011-01-01",
        "validation_start": "2001-01-01",
        "validation_end": "2006-01-01",
    },
    {
        "test": "2011-2015",
        "validation": "2006-2010",
        "test_start": "2011-01-01",
        "test_end": "2016-01-01",
        "validation_start": "2006-01-01",
        "validation_end": "2011-01-01",
    },
    {
        "test": "2016-2020",
        "validation": "2011-2015",
        "test_start": "2016-01-01",
        "test_end": "2021-01-01",
        "validation_start": "2011-01-01",
        "validation_end": "2016-01-01",
    },
    {
        "test": "2021-2025",
        "validation": "2016-2020",
        "test_start": "2021-01-01",
        "test_end": "2026-01-01",
        "validation_start": "2016-01-01",
        "validation_end": "2021-01-01",
    },
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("NESTED CHRONOLOGICAL DROUGHT-ONSET FEATURE SELECTION")
print("=" * 75)

df = pd.read_csv(INPUT)

df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

required_columns = [
    DATE_COL,
    SWBA_COL,
    "precipitation_mm",
    "pet_mm",
    "volumetric_soil_water_layer_1",
    "water_balance_3month",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df[DATE_COL].duplicated().any():
    raise ValueError("Duplicate monthly dates detected.")

print(f"Dataset rows: {len(df)}")
print(f"Date range: {df[DATE_COL].min().date()} to "
      f"{df[DATE_COL].max().date()}")


# ============================================================
# BUILD ONE-MONTH-AHEAD ONSET SAMPLES
# ============================================================

origin = df[
    [
        DATE_COL,
        SWBA_COL,
        "precipitation_mm",
        "pet_mm",
        "volumetric_soil_water_layer_1",
        "water_balance_3month",
    ]
].copy()

origin = origin.rename(
    columns={
        DATE_COL: "forecast_origin",
        SWBA_COL: "origin_swba",
        "precipitation_mm": "precipitation_mm",
        "pet_mm": "pet_mm",
        "volumetric_soil_water_layer_1":
            "volumetric_soil_water_layer_1",
        "water_balance_3month": "water_balance_3month",
    }
)

target = df[
    [DATE_COL, SWBA_COL]
].copy()

target = target.rename(
    columns={
        DATE_COL: "target_date",
        SWBA_COL: "target_swba",
    }
)

target["forecast_origin"] = (
    target["target_date"] - pd.DateOffset(months=1)
)

samples = target.merge(
    origin,
    on="forecast_origin",
    how="inner",
    validate="one_to_one",
)

# An onset occurs when the target month crosses into drought.
samples["onset_label"] = (
    (samples["target_swba"] <= DROUGHT_THRESHOLD) &
    (samples["origin_swba"] > DROUGHT_THRESHOLD)
).astype(int)

# Only retain genuine advance-warning opportunities.
samples = samples[
    samples["origin_swba"] > DROUGHT_THRESHOLD
].copy()

samples = samples.dropna(
    subset=[
        "precipitation_mm",
        "pet_mm",
        "volumetric_soil_water_layer_1",
        "water_balance_3month",
    ]
).reset_index(drop=True)

print(f"\nEligible observations: {len(samples)}")
print(f"Eligible onset cases: {samples['onset_label'].sum()}")


# ============================================================
# FOLD-SPECIFIC CALENDAR-MONTH ANOMALIES
# ============================================================

RAW_FEATURES = [
    "precipitation_mm",
    "pet_mm",
    "volumetric_soil_water_layer_1",
    "water_balance_3month",
]


def transform_anomalies(sample_data, reference_cutoff, features):
    """
    Calculate calendar-month anomalies using only reference
    observations available by the specified cutoff.
    """

    reference = df[
        df[DATE_COL] <= reference_cutoff
    ].copy()

    if reference.empty:
        raise ValueError(
            f"No reference data available by {reference_cutoff}"
        )

    reference["calendar_month"] = (
        reference[DATE_COL].dt.month
    )

    result = sample_data.copy()
    result["calendar_month"] = (
        result["forecast_origin"].dt.month
    )

    for feature in features:
        monthly_stats = reference.groupby(
            "calendar_month"
        )[feature].agg(["mean", "std"])

        means = result["calendar_month"].map(
            monthly_stats["mean"]
        )

        stds = result["calendar_month"].map(
            monthly_stats["std"]
        )

        stds = stds.replace(0, np.nan)

        result[feature + "_anom"] = (
            result[feature] - means
        ) / stds

    anomaly_features = [
        feature + "_anom" for feature in features
    ]

    result[anomaly_features] = (
        result[anomaly_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return result


# ============================================================
# MODEL
# ============================================================

def build_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def calculate_metrics(y_true, probabilities):
    predictions = (
        probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "test_samples": len(y_true),
        "observed_onsets": int(np.sum(y_true)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
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
        ),
    }


# ============================================================
# NESTED FEATURE SELECTION
# ============================================================

validation_results = []
test_results = []
prediction_records = []

for fold in FOLDS:

    test_name = fold["test"]
    validation_name = fold["validation"]

    validation_start = pd.Timestamp(
        fold["validation_start"]
    )
    validation_end = pd.Timestamp(
        fold["validation_end"]
    )

    test_start = pd.Timestamp(
        fold["test_start"]
    )
    test_end = pd.Timestamp(
        fold["test_end"]
    )

    validation_cutoff = (
        validation_start - pd.DateOffset(months=1)
    )

    test_cutoff = (
        test_start - pd.DateOffset(months=1)
    )

    print("\n" + "=" * 75)
    print(f"OUTER TEST PERIOD: {test_name}")
    print(f"FEATURE-SELECTION VALIDATION: {validation_name}")
    print("=" * 75)

    # --------------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------------

    validation_train = samples[
        samples["target_date"] < validation_start
    ].copy()

    validation_data = samples[
        (samples["target_date"] >= validation_start) &
        (samples["target_date"] < validation_end)
    ].copy()

    if validation_train["onset_label"].nunique() < 2:
        raise ValueError(
            f"Validation training data lacks both classes: "
            f"{test_name}"
        )

    if validation_data["onset_label"].nunique() < 2:
        raise ValueError(
            f"Validation period lacks both classes: "
            f"{validation_name}"
        )

    print(
        f"Validation training: {len(validation_train)} "
        f"observations, "
        f"{validation_train['onset_label'].sum()} onsets"
    )

    print(
        f"Validation evaluation: {len(validation_data)} "
        f"observations, "
        f"{validation_data['onset_label'].sum()} onsets"
    )

    candidate_scores = []

    # --------------------------------------------------------
    # COMPARE ALL FEATURE SETS ON VALIDATION DATA
    # --------------------------------------------------------

    for feature_name, raw_features in FEATURE_SETS.items():

        anomaly_features = [
            feature + "_anom" for feature in raw_features
        ]

        train_transformed = transform_anomalies(
            validation_train,
            validation_cutoff,
            raw_features,
        )

        validation_transformed = transform_anomalies(
            validation_data,
            validation_cutoff,
            raw_features,
        )

        model = build_model()

        model.fit(
            train_transformed[anomaly_features],
            train_transformed["onset_label"],
        )

        probabilities = model.predict_proba(
            validation_transformed[anomaly_features]
        )[:, 1]

        score = average_precision_score(
            validation_transformed["onset_label"],
            probabilities,
        )

        metrics = calculate_metrics(
            validation_transformed["onset_label"].to_numpy(),
            probabilities,
        )

        candidate_scores.append({
            "feature_set": feature_name,
            "predictors": ", ".join(raw_features),
            "predictor_count": len(raw_features),
            "validation_pr_auc": score,
            "validation_precision": metrics["precision"],
            "validation_recall": metrics["recall"],
            "validation_f1": metrics["f1"],
            "validation_TP": metrics["TP"],
            "validation_FP": metrics["FP"],
            "validation_FN": metrics["FN"],
            "validation_TN": metrics["TN"],
        })

        print(
            f"{feature_name:25s} "
            f"PR-AUC={score:.3f} "
            f"Recall={metrics['recall']:.3f} "
            f"Precision={metrics['precision']:.3f}"
        )

    # Highest validation PR-AUC wins.
    # If tied, prefer the smaller feature set.
    candidate_scores.sort(
        key=lambda row: (
            -row["validation_pr_auc"],
            row["predictor_count"],
            row["feature_set"],
        )
    )

    selected = candidate_scores[0]
    selected_name = selected["feature_set"]
    selected_features = FEATURE_SETS[selected_name]

    print("\nSELECTED FEATURE SET:")
    print(selected_name)
    print("Predictors:", ", ".join(selected_features))
    print(
        f"Validation PR-AUC: "
        f"{selected['validation_pr_auc']:.4f}"
    )

    for row in candidate_scores:
        row["outer_test_period"] = test_name
        row["validation_period"] = validation_name
        row["selected"] = (
            row["feature_set"] == selected_name
        )
        validation_results.append(row)

    # --------------------------------------------------------
    # FINAL OUTER TEST
    # --------------------------------------------------------

    final_train = samples[
        samples["target_date"] < test_start
    ].copy()

    test_data = samples[
        (samples["target_date"] >= test_start) &
        (samples["target_date"] < test_end)
    ].copy()

    if final_train["onset_label"].nunique() < 2:
        raise ValueError(
            f"Final training data lacks both classes: {test_name}"
        )

    print(
        f"\nFinal training: {len(final_train)} observations, "
        f"{final_train['onset_label'].sum()} onsets"
    )

    print(
        f"Outer test: {len(test_data)} observations, "
        f"{test_data['onset_label'].sum()} onsets"
    )

    # Evaluate selected feature set and fixed four-feature benchmark.
    models_to_test = {
        "nested_selected": selected_features,
        "fixed_full_four": FEATURE_SETS["full_four_predictors"],
    }

    for model_label, raw_features in models_to_test.items():

        anomaly_features = [
            feature + "_anom" for feature in raw_features
        ]

        train_transformed = transform_anomalies(
            final_train,
            test_cutoff,
            raw_features,
        )

        test_transformed = transform_anomalies(
            test_data,
            test_cutoff,
            raw_features,
        )

        model = build_model()

        model.fit(
            train_transformed[anomaly_features],
            train_transformed["onset_label"],
        )

        probabilities = model.predict_proba(
            test_transformed[anomaly_features]
        )[:, 1]

        metrics = calculate_metrics(
            test_transformed["onset_label"].to_numpy(),
            probabilities,
        )

        result = {
            "test_period": test_name,
            "validation_period": validation_name,
            "model": model_label,
            "selected_feature_set": selected_name,
            "predictors": ", ".join(raw_features),
            **metrics,
        }

        test_results.append(result)

        print(
            f"\n{model_label}: "
            f"PR-AUC={metrics['pr_auc']:.3f}, "
            f"Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}, "
            f"FP={metrics['FP']}"
        )

        for i, (_, sample) in enumerate(
            test_transformed.iterrows()
        ):
            prediction_records.append({
                "test_period": test_name,
                "model": model_label,
                "target_date": sample["target_date"],
                "forecast_origin": sample["forecast_origin"],
                "actual_onset": int(sample["onset_label"]),
                "predicted_probability": float(probabilities[i]),
                "predicted_onset": int(
                    probabilities[i] >= CLASSIFICATION_THRESHOLD
                ),
                "selected_feature_set": selected_name,
            })


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

validation_df = pd.DataFrame(validation_results)
test_df = pd.DataFrame(test_results)
predictions_df = pd.DataFrame(prediction_records)

validation_path = (
    OUTPUT / "nested_validation_feature_scores.csv"
)

test_path = (
    OUTPUT / "nested_outer_test_fold_metrics.csv"
)

predictions_path = (
    OUTPUT / "nested_outer_test_predictions.csv"
)

validation_df.to_csv(validation_path, index=False)
test_df.to_csv(test_path, index=False)
predictions_df.to_csv(predictions_path, index=False)


# ============================================================
# POOLED OUT-OF-SAMPLE RESULTS
# ============================================================

print("\n" + "=" * 75)
print("POOLED OUT-OF-SAMPLE RESULTS")
print("=" * 75)

pooled_results = []

for model_label, group in predictions_df.groupby("model"):

    metrics = calculate_metrics(
        group["actual_onset"].to_numpy(),
        group["predicted_probability"].to_numpy(),
    )

    pooled_results.append({
        "model": model_label,
        **metrics,
    })

pooled_df = pd.DataFrame(pooled_results)

pooled_path = OUTPUT / "nested_pooled_metrics.csv"
pooled_df.to_csv(pooled_path, index=False)

print(pooled_df.to_string(index=False))


# ============================================================
# SELECTION SUMMARY
# ============================================================

selection_summary = (
    validation_df[validation_df["selected"]]
    [
        [
            "outer_test_period",
            "validation_period",
            "feature_set",
            "predictors",
            "validation_pr_auc",
        ]
    ]
    .sort_values("outer_test_period")
)

selection_path = OUTPUT / "selected_feature_sets_by_fold.csv"
selection_summary.to_csv(selection_path, index=False)

print("\n" + "=" * 75)
print("FEATURE SET SELECTED IN EACH OUTER FOLD")
print("=" * 75)

print(selection_summary.to_string(index=False))

print("\nSaved results to:")
print(OUTPUT)

print("\nExperiment complete.")
