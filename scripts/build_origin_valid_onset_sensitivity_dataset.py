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
    / "origin_valid_sensitivity"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = -1.0
MIN_HISTORY = 15
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

# ============================================================
# LOAD AND VALIDATE HISTORICAL DATA
# ============================================================

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

df = df.set_index("period")
df.index.name = "target_period"

if df.index.duplicated().any():
    raise ValueError("Duplicate monthly periods detected.")

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Missing predictor columns: {missing}")

swba_lookup = df["swba_3month"].to_dict()

previous_swba = pd.Series(
    (df.index - 1).map(swba_lookup),
    index=df.index
)

df["drought_onset"] = (
    df["swba_3month"].notna()
    & (df["swba_3month"] <= THRESHOLD)
    & previous_swba.notna()
    & (previous_swba > THRESHOLD)
)

# ============================================================
# BUILD ORIGIN-VALID ANOMALIES
# ============================================================

all_samples = []
audit_results = []

print("=" * 75)
print("ORIGIN-VALID DROUGHT-ONSET SENSITIVITY DATASET")
print("=" * 75)

print(f"Minimum historical observations: {MIN_HISTORY}")
print("Reference: same calendar month, strictly before origin")
print("Standard deviation: population (ddof=0)")

for lead in LEADS:

    print("\n" + "=" * 75)
    print(f"LEAD: {lead} MONTH(S)")
    print("=" * 75)

    base = df[
        ["swba_3month", "drought_onset"]
    ].dropna(subset=["swba_3month"]).copy()

    base["forecast_origin"] = base.index - lead

    base["origin_swba"] = base["forecast_origin"].map(
        swba_lookup
    )

    base = base.dropna(subset=["origin_swba"])
    base = base[
        base["origin_swba"] > THRESHOLD
    ].copy()

    base["onset_label"] = (
        base["drought_onset"].astype(int)
    )

    # Attach raw predictor values from the forecast origin.
    for feature in FEATURES:
        base[feature] = base["forecast_origin"].map(
            df[feature]
        )

    valid_records = []

    # Calculate each observation's reference using only
    # complete historical records strictly before its origin.
    for target_period, row in base.iterrows():

        origin = row["forecast_origin"]

        current_values = row[FEATURES]

        if current_values.isna().any():
            continue

        history_mask = (
            (df.index < origin)
            & (df["calendar_month"] == origin.month)
            & df[FEATURES].notna().all(axis=1)
        )

        history = df.loc[history_mask, FEATURES]

        if len(history) < MIN_HISTORY:
            continue

        monthly_means = history.mean()
        monthly_stds = history.std(ddof=0)

        if (
            monthly_stds.isna().any()
            or (monthly_stds <= 0).any()
        ):
            continue

        record = {
            "target_period": target_period,
            "swba_3month": row["swba_3month"],
            "drought_onset": bool(row["drought_onset"]),
            "forecast_origin": origin,
            "origin_swba": row["origin_swba"],
            "onset_label": int(row["onset_label"]),
            "historical_reference_n": len(history),
        }

        for feature in FEATURES:

            record[feature] = current_values[feature]

            record[f"{feature}_anom"] = (
                current_values[feature]
                - monthly_means[feature]
            ) / monthly_stds[feature]

        valid_records.append(record)

    if not valid_records:
        raise ValueError(
            f"No eligible observations for lead {lead}."
        )

    valid = pd.DataFrame(valid_records)
    valid["target_period"] = pd.PeriodIndex(
        valid["target_period"], freq="M"
    )
    valid = valid.set_index("target_period").sort_index()

    anomaly_features = [
        f"{feature}_anom" for feature in FEATURES
    ]

    # ========================================================
    # CREATE CHRONOLOGICAL FOLDS
    # ========================================================

    for test_start, test_end in FOLDS:

        first_test_target = pd.Period(
            f"{test_start}-01", freq="M"
        )

        cutoff = first_test_target - lead

        train = valid[valid.index <= cutoff].copy()

        test = valid[
            (valid.index.year >= test_start)
            & (valid.index.year <= test_end)
        ].copy()

        train["split"] = "train"
        test["split"] = "test"

        combined = pd.concat([train, test]).copy()

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

        combined = combined[
            combined["complete_predictors"]
        ].copy()

        train_complete = combined[
            combined["split"] == "train"
        ]

        test_complete = combined[
            combined["split"] == "test"
        ]

        train_onsets = int(train_complete["onset_label"].sum())
        test_onsets = int(test_complete["onset_label"].sum())

        print(f"\nFold: {test_start}-{test_end}")
        print(
            f"  Training: {len(train_complete)} observations, "
            f"{train_onsets} onsets"
        )
        print(
            f"  Testing: {len(test_complete)} observations, "
            f"{test_onsets} onsets"
        )

        if train_complete["onset_label"].nunique() < 2:
            raise ValueError(
                f"Training set has only one class: "
                f"lead={lead}, fold={test_start}-{test_end}"
            )

        if test_complete.empty:
            raise ValueError(
                f"Empty test set: lead={lead}, "
                f"fold={test_start}-{test_end}"
            )

        audit_results.append({
            "lead_months": lead,
            "test_period": f"{test_start}-{test_end}",
            "training_samples": len(train_complete),
            "training_onsets": train_onsets,
            "test_samples": len(test_complete),
            "test_onsets": test_onsets,
            "minimum_history": MIN_HISTORY,
        })

        all_samples.append(
            combined.reset_index(drop=True)
        )

# ============================================================
# SAVE SEPARATE SENSITIVITY OUTPUTS
# ============================================================

final = pd.concat(all_samples, ignore_index=True)

dataset_path = (
    OUTPUT_DIR / "origin_valid_onset_fold_dataset.csv"
)

audit_path = (
    OUTPUT_DIR / "origin_valid_sample_size_audit.csv"
)

final.to_csv(dataset_path, index=False)

audit = pd.DataFrame(audit_results)
audit.to_csv(audit_path, index=False)

print("\n" + "=" * 75)
print("SENSITIVITY DATASET CONSTRUCTION COMPLETE")
print("=" * 75)

print("Total rows:", len(final))
print(
    "Training rows:",
    int((final["split"] == "train").sum())
)
print(
    "Test rows:",
    int((final["split"] == "test").sum())
)

print("\nSaved:")
print(dataset_path)
print(audit_path)

print("\nOriginal dataset and model outputs were not modified.")
