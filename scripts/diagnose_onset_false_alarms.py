from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "outputs" / "expanded" /
    "drought_onset_model_preparation" /
    "nested_feature_selection" /
    "nested_outer_test_predictions.csv"
)

OUTPUT = (
    ROOT / "outputs" / "expanded" /
    "drought_onset_model_preparation" /
    "false_alarm_diagnostics"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5

print("=" * 75)
print("DROUGHT-ONSET FALSE-ALARM AND WARNING DIAGNOSTICS")
print("=" * 75)

# ------------------------------------------------------------
# LOAD FIXED FOUR-PREDICTOR OUT-OF-SAMPLE PREDICTIONS
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

df = df[
    df["model"] == "fixed_full_four"
].copy()

df["target_date"] = pd.to_datetime(df["target_date"])
df["forecast_origin"] = pd.to_datetime(df["forecast_origin"])

df = df.sort_values("target_date").reset_index(drop=True)

if df.empty:
    raise ValueError("No fixed_full_four predictions found.")

if df["target_date"].duplicated().any():
    raise ValueError("Duplicate target dates found.")

df["actual_onset"] = df["actual_onset"].astype(int)

df["predicted_onset"] = (
    df["predicted_probability"] >= THRESHOLD
).astype(int)

# Classification categories
df["classification"] = np.select(
    [
        (df["actual_onset"] == 1) & (df["predicted_onset"] == 1),
        (df["actual_onset"] == 0) & (df["predicted_onset"] == 1),
        (df["actual_onset"] == 1) & (df["predicted_onset"] == 0),
        (df["actual_onset"] == 0) & (df["predicted_onset"] == 0),
    ],
    ["TP", "FP", "FN", "TN"],
    default="Unknown",
)

df["target_month"] = df["target_date"].dt.month
df["origin_month"] = df["forecast_origin"].dt.month

# Southern Hemisphere meteorological seasons
season_map = {
    12: "Summer",
    1: "Summer",
    2: "Summer",
    3: "Autumn",
    4: "Autumn",
    5: "Autumn",
    6: "Winter",
    7: "Winter",
    8: "Winter",
    9: "Spring",
    10: "Spring",
    11: "Spring",
}

df["target_season"] = df["target_month"].map(season_map)
df["origin_season"] = df["origin_month"].map(season_map)

# ------------------------------------------------------------
# OVERALL CLASSIFICATION SUMMARY
# ------------------------------------------------------------

print("\n1. OVERALL CLASSIFICATION")

print(
    df["classification"]
    .value_counts()
    .reindex(["TP", "FP", "FN", "TN"], fill_value=0)
    .to_string()
)

print(f"\nTotal predictions: {len(df)}")
print(f"Observed onset events: {df['actual_onset'].sum()}")
print(f"Predicted warnings: {df['predicted_onset'].sum()}")
print(f"Classification threshold: {THRESHOLD}")

# ------------------------------------------------------------
# MONTHLY FALSE-ALARM SUMMARY
# ------------------------------------------------------------

def grouped_summary(data, group_column):
    rows = []

    for group_value, group in data.groupby(
        group_column, sort=True
    ):
        tp = int((group["classification"] == "TP").sum())
        fp = int((group["classification"] == "FP").sum())
        fn = int((group["classification"] == "FN").sum())
        tn = int((group["classification"] == "TN").sum())

        warnings = tp + fp

        rows.append({
            group_column: group_value,
            "observations": len(group),
            "observed_onsets": tp + fn,
            "predicted_warnings": warnings,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": tp / warnings if warnings else np.nan,
            "false_alarm_rate": (
                fp / (fp + tn) if (fp + tn) else np.nan
            ),
        })

    return pd.DataFrame(rows)


target_month_summary = grouped_summary(
    df, "target_month"
)

origin_month_summary = grouped_summary(
    df, "origin_month"
)

season_summary = grouped_summary(
    df, "target_season"
)

target_month_summary.to_csv(
    OUTPUT / "false_alarms_by_target_month.csv",
    index=False,
)

origin_month_summary.to_csv(
    OUTPUT / "false_alarms_by_forecast_origin_month.csv",
    index=False,
)

season_summary.to_csv(
    OUTPUT / "false_alarms_by_target_season.csv",
    index=False,
)

print("\n2. FALSE ALARMS BY TARGET MONTH")
print(target_month_summary.to_string(index=False))

print("\n3. FALSE ALARMS BY FORECAST ORIGIN MONTH")
print(origin_month_summary.to_string(index=False))

print("\n4. FALSE ALARMS BY TARGET SEASON")
print(season_summary.to_string(index=False))

# ------------------------------------------------------------
# PROBABILITY DISTRIBUTION BY CLASSIFICATION CATEGORY
# ------------------------------------------------------------

probability_summary = (
    df.groupby("classification")["predicted_probability"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        minimum="min",
        maximum="max",
        standard_deviation="std",
    )
    .reindex(["TP", "FP", "FN", "TN"])
    .reset_index()
)

probability_summary.to_csv(
    OUTPUT / "probability_by_classification.csv",
    index=False,
)

print("\n5. PREDICTED PROBABILITIES BY CLASSIFICATION")
print(probability_summary.to_string(index=False))

# ------------------------------------------------------------
# FALSE ALARM RECORDS
# ------------------------------------------------------------

false_alarms = df[
    df["classification"] == "FP"
].copy()

false_alarms = false_alarms[
    [
        "test_period",
        "forecast_origin",
        "target_date",
        "predicted_probability",
        "classification",
    ]
].sort_values("target_date")

false_alarms.to_csv(
    OUTPUT / "individual_false_alarm_months.csv",
    index=False,
)

print("\n6. INDIVIDUAL FALSE ALARM MONTHS")
print(false_alarms.to_string(index=False))

# ------------------------------------------------------------
# DETECTED AND MISSED ONSET EVENTS
# ------------------------------------------------------------

onset_events = df[
    df["actual_onset"] == 1
].copy()

onset_events = onset_events[
    [
        "test_period",
        "forecast_origin",
        "target_date",
        "predicted_probability",
        "predicted_onset",
        "classification",
    ]
].sort_values("target_date")

onset_events.to_csv(
    OUTPUT / "individual_onset_event_predictions.csv",
    index=False,
)

print("\n7. ALL OBSERVED ONSET EVENTS")
print(onset_events.to_string(index=False))

# ------------------------------------------------------------
# PROBABILITY BANDS
# ------------------------------------------------------------

df["probability_band"] = pd.cut(
    df["predicted_probability"],
    bins=[-np.inf, 0.1, 0.25, 0.5, 0.75, 1.0],
    labels=[
        "Below 0.10",
        "0.10 to 0.25",
        "0.25 to 0.50",
        "0.50 to 0.75",
        "0.75 to 1.00",
    ],
    right=False,
)

probability_band_summary = (
    df.groupby("probability_band", observed=False)
    .agg(
        observations=("actual_onset", "size"),
        observed_onsets=("actual_onset", "sum"),
        predicted_warnings=("predicted_onset", "sum"),
    )
    .reset_index()
)

probability_band_summary.to_csv(
    OUTPUT / "probability_band_summary.csv",
    index=False,
)

print("\n8. PROBABILITY BANDS")
print(probability_band_summary.to_string(index=False))

# ------------------------------------------------------------
# SAVE COMPLETE CLASSIFIED DATA
# ------------------------------------------------------------

df.to_csv(
    OUTPUT / "classified_out_of_sample_predictions.csv",
    index=False,
)

print("\n" + "=" * 75)
print("DIAGNOSTIC COMPLETE")
print("=" * 75)

print(f"Results saved to:\n{OUTPUT}")
