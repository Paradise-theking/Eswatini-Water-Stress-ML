from pathlib import Path
import pandas as pd
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT / "data" / "processed" / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)

OUTPUT_DIR = (
    PROJECT / "outputs" / "expanded"
    / "drought_onset_model_preparation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = -1.0
LEADS = [1, 2, 3]

FOLDS = [
    (2006, 2010),
    (2011, 2015),
    (2016, 2020),
    (2021, 2025),
]

FEATURES = [
    "precipitation_mm",
    "pet_mm",
    "volumetric_soil_water_layer_1",
    "water_balance_3month",
]

df = pd.read_csv(INPUT_FILE)

df["month_date"] = pd.to_datetime(df["month_date"])
df["swba_3month"] = pd.to_numeric(
    df["swba_3month"], errors="coerce"
)

df = (
    df.sort_values("month_date")
    .drop_duplicates("month_date")
    .reset_index(drop=True)
)

df["period"] = df["month_date"].dt.to_period("M")
df["calendar_month"] = df["month_date"].dt.month

if df["period"].duplicated().any():
    raise ValueError("Duplicate monthly periods detected.")

df = df.set_index("period")
df.index.name = "target_period"

swba_lookup = df["swba_3month"].to_dict()

previous_swba = (
    (df.index - 1).map(swba_lookup)
)

df["drought_onset"] = (
    df["swba_3month"].notna()
    & (df["swba_3month"] <= THRESHOLD)
    & pd.Series(previous_swba, index=df.index).notna()
    & (pd.Series(previous_swba, index=df.index) > THRESHOLD)
)

print("=" * 75)
print("DROUGHT-ONSET MODELLING DATASET CONSTRUCTION")
print("=" * 75)

print("\nCandidate predictors:")
for feature in FEATURES:
    if feature not in df.columns:
        raise ValueError(f"Missing predictor column: {feature}")
    print(f"  {feature}")

all_samples = []

for lead in LEADS:

    print("\n" + "=" * 75)
    print(f"LEAD: {lead} MONTH(S)")
    print("=" * 75)

    base = df[
        ["swba_3month", "drought_onset"]
    ].copy()

    base = base.dropna(subset=["swba_3month"])

    base["forecast_origin"] = base.index - lead
    base["origin_swba"] = base["forecast_origin"].map(
        swba_lookup
    )

    # Require the catchment to be outside drought at origin.
    base = base.dropna(subset=["origin_swba"])
    base = base[
        base["origin_swba"] > THRESHOLD
    ].copy()

    base["onset_label"] = (
        base["drought_onset"].astype(int)
    )

    # Attach predictor values from forecast-origin month.
    for feature in FEATURES:
        feature_lookup = df[feature].to_dict()
        base[feature] = base["forecast_origin"].map(
            feature_lookup
        )

    for test_start, test_end in FOLDS:

        first_test_target = pd.Period(
            f"{test_start}-01", freq="M"
        )

        cutoff = first_test_target - lead

        train = base[base.index <= cutoff].copy()

        test = base[
            (base.index.year >= test_start)
            & (base.index.year <= test_end)
        ].copy()

        # Fit month-of-year reference statistics using
        # information available by the fold's training cutoff.
        reference = df.loc[
            df.index <= cutoff,
            FEATURES + ["calendar_month"]
        ].copy()

        monthly_means = reference.groupby(
            "calendar_month"
        )[FEATURES].mean()

        monthly_stds = reference.groupby(
            "calendar_month"
        )[FEATURES].std(ddof=0)

        if monthly_means.isna().any().any():
            raise ValueError(
                f"Missing monthly means for lead {lead}, "
                f"fold {test_start}-{test_end}"
            )

        if (monthly_stds <= 0).any().any() or monthly_stds.isna().any().any():
            raise ValueError(
                f"Invalid monthly standard deviations for "
                f"lead {lead}, fold {test_start}-{test_end}"
            )

        # Transform origin-month predictors using the
        # fold-specific historical calendar-month reference.
        for feature in FEATURES:

            origin_month = (
                base["forecast_origin"].map(
                    df["calendar_month"].to_dict()
                )
            )

            means = origin_month.map(
                monthly_means[feature]
            )

            stds = origin_month.map(
                monthly_stds[feature]
            )

            base[f"{feature}_anom"] = (
                base[feature] - means
            ) / stds

        anomaly_features = [
            f"{feature}_anom" for feature in FEATURES
        ]

        train_fold = base[
            base.index <= cutoff
        ].copy()

        test_fold = base[
            (base.index.year >= test_start)
            & (base.index.year <= test_end)
        ].copy()

        train_fold["split"] = "train"
        test_fold["split"] = "test"

        combined = pd.concat(
            [train_fold, test_fold]
        ).copy()

        combined["lead_months"] = lead
        combined["test_period"] = (
            f"{test_start}-{test_end}"
        )

        combined["target_period_text"] = (
            combined.index.astype(str)
        )

        combined["forecast_origin_text"] = (
            combined["forecast_origin"].astype(str)
        )

        combined["complete_predictors"] = (
            combined[anomaly_features]
            .notna()
            .all(axis=1)
        )

        # Retain complete cases for the eventual model.
        complete = combined[
            combined["complete_predictors"]
        ].copy()

        train_complete = complete[
            complete["split"] == "train"
        ]

        test_complete = complete[
            complete["split"] == "test"
        ]

        print(f"\nFold: {test_start}-{test_end}")
        print(
            f"  Training rows before predictor check: "
            f"{len(train_fold)}"
        )
        print(
            f"  Training rows with complete predictors: "
            f"{len(train_complete)}"
        )
        print(
            f"  Training onset cases: "
            f"{int(train_complete['onset_label'].sum())}"
        )
        print(
            f"  Test rows before predictor check: "
            f"{len(test_fold)}"
        )
        print(
            f"  Test rows with complete predictors: "
            f"{len(test_complete)}"
        )
        print(
            f"  Test onset cases: "
            f"{int(test_complete['onset_label'].sum())}"
        )

        all_samples.append(
            complete.reset_index(drop=True)
        )

# ============================================================
# SAVE DATASET
# ============================================================

final = pd.concat(
    all_samples,
    ignore_index=True
)

output_file = (
    OUTPUT_DIR
    / "onset_classification_fold_dataset.csv"
)

final.to_csv(output_file, index=False)

print("\n" + "=" * 75)
print("DATASET CONSTRUCTION COMPLETE")
print("=" * 75)

print("Total fold-specific sample rows:", len(final))
print("Total training rows:", int((final["split"] == "train").sum()))
print("Total test rows:", int((final["split"] == "test").sum()))

print("\nPredictor missingness after filtering:")
for feature in FEATURES:
    col = f"{feature}_anom"
    print(f"  {col}: {final[col].isna().sum()}")

print("\nSaved:")
print(output_file)

print("\nNo models have been trained.")
