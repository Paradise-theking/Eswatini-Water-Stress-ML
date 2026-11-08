from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

INPUT = Path(
    "data/processed/expanded/"
    "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)

OUT = Path(
    "outputs/expanded/drought_onset_model_preparation/"
    "feature_ablation"
)
OUT.mkdir(parents=True, exist_ok=True)

LEAD = 1
THRESHOLD = 0.50

FEATURES = {
    "rainfall": ["precipitation_mm"],
    "PET": ["pet_mm"],
    "soil_moisture": ["volumetric_soil_water_layer_1"],
    "water_balance": ["water_balance_3month"],
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
    ]
}

TEST_PERIODS = [
    (2006, 2010),
    (2011, 2015),
    (2016, 2020),
    (2021, 2025)
]

# ------------------------------------------------------------
# LOAD DATA AND DEFINE ONSET
# ------------------------------------------------------------

df = pd.read_csv(INPUT)
df["month_date"] = pd.to_datetime(df["month_date"])
df = df.sort_values("month_date").reset_index(drop=True)

df["previous_swba"] = df["swba_3month"].shift(1)

df["onset_label"] = (
    (df["swba_3month"] <= -1)
    & (df["previous_swba"] > -1)
).astype(int)

df.loc[df["previous_swba"].isna(), "onset_label"] = np.nan

origin = df[
    ["month_date", "swba_3month"] +
    list(FEATURES["full_four_predictors"])
].copy()

origin = origin.rename(columns={
    "month_date": "forecast_origin",
    "swba_3month": "origin_swba"
})

targets = df[
    ["month_date", "swba_3month", "onset_label"]
].dropna().copy()

targets["forecast_origin"] = (
    targets["month_date"] - pd.DateOffset(months=LEAD)
)

samples = targets.merge(
    origin,
    on="forecast_origin",
    how="inner"
)

samples = samples[
    samples["origin_swba"] > -1
].copy()

samples["origin_month"] = samples["forecast_origin"].dt.month
samples["onset_label"] = samples["onset_label"].astype(int)

print("=" * 75)
print("ONE-MONTH DROUGHT-ONSET FEATURE-ABLATION EXPERIMENT")
print("=" * 75)

print("Eligible observations:", len(samples))
print("Eligible onset cases:", int(samples["onset_label"].sum()))

# ------------------------------------------------------------
# FOLD-SPECIFIC ANOMALIES
# ------------------------------------------------------------

def transform(data, cutoff, raw_features):

    reference = df[df["month_date"] <= cutoff].copy()
    reference["calendar_month"] = reference["month_date"].dt.month

    result = data.copy()

    for feature in raw_features:

        stats = reference.groupby("calendar_month")[feature].agg(
            mean="mean",
            std=lambda x: x.std(ddof=0)
        )

        months = result["origin_month"]

        means = months.map(stats["mean"])
        stds = months.map(stats["std"])

        if means.isna().any() or stds.isna().any():
            raise ValueError(
                f"Missing monthly reference statistics for {feature}"
            )

        if (stds <= 0).any():
            raise ValueError(
                f"Non-positive standard deviation for {feature}"
            )

        result[feature + "_anom"] = (
            result[feature].to_numpy() - means.to_numpy()
        ) / stds.to_numpy()

    return result


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

def make_model():

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


# ------------------------------------------------------------
# RUN ALL FEATURE COMBINATIONS
# ------------------------------------------------------------

all_predictions = []
fold_metrics = []

for combination, raw_features in FEATURES.items():

    print("\n" + "-" * 75)
    print("FEATURE SET:", combination)
    print("Predictors:", ", ".join(raw_features))

    combination_predictions = []

    for start, end in TEST_PERIODS:

        first_test_target = pd.Timestamp(start, 1, 1)

        cutoff = (
            first_test_target
            - pd.DateOffset(months=LEAD)
        )

        train = samples[
            samples["month_date"] <= cutoff
        ].copy()

        test = samples[
            samples["month_date"].dt.year.between(start, end)
        ].copy()

        if train["onset_label"].nunique() < 2:
            raise ValueError(
                f"Training data has one class: {combination}, {start}-{end}"
            )

        train_t = transform(train, cutoff, raw_features)
        test_t = transform(test, cutoff, raw_features)

        model_features = [
            feature + "_anom"
            for feature in raw_features
        ]

        model = make_model()

        model.fit(
            train_t[model_features],
            train_t["onset_label"]
        )

        probabilities = model.predict_proba(
            test_t[model_features]
        )[:, 1]

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        y_true = test_t["onset_label"].to_numpy()

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()

        metrics = {
            "feature_set": combination,
            "test_period": f"{start}-{end}",
            "training_samples": len(train_t),
            "test_samples": len(test_t),
            "test_onsets": int(y_true.sum()),
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
            )
        }

        fold_metrics.append(metrics)

        for i in range(len(test_t)):

            combination_predictions.append({
                "feature_set": combination,
                "test_period": f"{start}-{end}",
                "forecast_origin": (
                    test_t.iloc[i]["forecast_origin"].strftime("%Y-%m")
                ),
                "target_month": (
                    test_t.iloc[i]["month_date"].strftime("%Y-%m")
                ),
                "actual_onset": int(y_true[i]),
                "probability": float(probabilities[i]),
                "predicted_onset": int(predictions[i])
            })

        print(
            f"{start}-{end}: "
            f"onsets={int(y_true.sum())}, "
            f"TP={tp}, FP={fp}, FN={fn}, TN={tn}, "
            f"PR-AUC={metrics['pr_auc']:.3f}"
        )

    all_predictions.extend(combination_predictions)


# ------------------------------------------------------------
# POOLED OUT-OF-SAMPLE METRICS
# ------------------------------------------------------------

pred_df = pd.DataFrame(all_predictions)
fold_df = pd.DataFrame(fold_metrics)

pooled = []

for combination in FEATURES:

    subset = pred_df[
        pred_df["feature_set"] == combination
    ]

    y_true = subset["actual_onset"].to_numpy()
    probabilities = subset["probability"].to_numpy()
    predictions = subset["predicted_onset"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    pooled.append({
        "feature_set": combination,
        "test_samples": len(subset),
        "observed_onsets": int(y_true.sum()),
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
        )
    })

pooled_df = pd.DataFrame(pooled).sort_values(
    "pr_auc",
    ascending=False
)

fold_df.to_csv(OUT / "feature_ablation_fold_metrics.csv", index=False)
pred_df.to_csv(OUT / "feature_ablation_predictions.csv", index=False)
pooled_df.to_csv(OUT / "feature_ablation_pooled_metrics.csv", index=False)

print("\n" + "=" * 75)
print("POOLED OUT-OF-SAMPLE FEATURE COMPARISON")
print("=" * 75)

print(pooled_df.round(4).to_string(index=False))

print("\nSaved results to:")
print(OUT.resolve())

print("\nExperiment complete.")
