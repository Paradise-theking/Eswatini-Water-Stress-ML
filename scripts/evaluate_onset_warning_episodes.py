from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# EVENT-BASED DROUGHT WARNING EPISODE EVALUATION
# ============================================================

INPUT = Path(
    "outputs/expanded/drought_onset_model_preparation/"
    "nested_feature_selection/nested_outer_test_predictions.csv"
)

OUTPUT_DIR = Path(
    "outputs/expanded/drought_onset_model_preparation/"
    "warning_episode_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "fixed_full_four"

# ============================================================
# LOAD AND VALIDATE PREDICTIONS
# ============================================================

df = pd.read_csv(INPUT)

required_columns = [
    "test_period",
    "model",
    "target_date",
    "forecast_origin",
    "actual_onset",
    "predicted_probability",
    "predicted_onset",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[df["model"] == MODEL].copy()

df["target_date"] = pd.to_datetime(df["target_date"])
df["forecast_origin"] = pd.to_datetime(df["forecast_origin"])

df["actual_onset"] = pd.to_numeric(
    df["actual_onset"]
).astype(int)

df["predicted_onset"] = pd.to_numeric(
    df["predicted_onset"]
).astype(int)

df["predicted_probability"] = pd.to_numeric(
    df["predicted_probability"]
)

df = df.sort_values(
    ["test_period", "target_date"]
).reset_index(drop=True)

if df.empty:
    raise ValueError("No fixed-model predictions found.")

if df.duplicated(["test_period", "target_date"]).any():
    raise ValueError("Duplicate target months detected.")

if not df["actual_onset"].isin([0, 1]).all():
    raise ValueError("Unexpected actual onset labels.")

if not df["predicted_onset"].isin([0, 1]).all():
    raise ValueError("Unexpected predicted onset labels.")

# Verify one-month forecast alignment.
expected_origin = (
    df["target_date"] - pd.DateOffset(months=1)
)

if not (df["forecast_origin"] == expected_origin).all():
    raise ValueError(
        "Forecast origin does not consistently precede "
        "the target by one month."
    )

print("=" * 70)
print("DROUGHT WARNING EPISODE EVALUATION")
print("=" * 70)

print(f"Model: {MODEL}")
print(f"Eligible test observations: {len(df)}")
print(f"Observed onset events: {df['actual_onset'].sum()}")
print(f"Predicted warning months: {df['predicted_onset'].sum()}")

# ============================================================
# IDENTIFY CONSECUTIVE WARNING EPISODES
# ============================================================

df["warning_episode_id"] = None

episode_records = []

episode_counter = 0

for test_period, group in df.groupby(
    "test_period",
    sort=True
):

    group = group.sort_values("target_date").copy()

    previous_date = None
    previous_warning = False
    current_episode_id = None

    for index, row in group.iterrows():

        is_warning = row["predicted_onset"] == 1

        current_date = row["target_date"]

        consecutive_month = False

        if previous_date is not None:
            month_difference = (
                (current_date.year - previous_date.year) * 12
                + current_date.month
                - previous_date.month
            )

            consecutive_month = month_difference == 1

        if is_warning:

            if not (previous_warning and consecutive_month):

                episode_counter += 1

                current_episode_id = (
                    f"{test_period}_E{episode_counter:03d}"
                )

            df.loc[index, "warning_episode_id"] = (
                current_episode_id
            )

        else:
            current_episode_id = None

        previous_date = current_date
        previous_warning = is_warning

# ============================================================
# SUMMARISE EACH WARNING EPISODE
# ============================================================

warnings = df[
    df["predicted_onset"] == 1
].copy()

for episode_id, group in warnings.groupby(
    "warning_episode_id",
    sort=True
):

    group = group.sort_values("target_date")

    warning_months = len(group)
    detected_onsets = int(group["actual_onset"].sum())
    false_warning_months = warning_months - detected_onsets

    if detected_onsets > 0 and false_warning_months > 0:
        episode_type = "mixed"

    elif detected_onsets > 0:
        episode_type = "hit_only"

    else:
        episode_type = "false_only"

    episode_records.append({
        "warning_episode_id": episode_id,
        "test_period": group["test_period"].iloc[0],
        "episode_start_target": group["target_date"].min(),
        "episode_end_target": group["target_date"].max(),
        "episode_start_origin": group["forecast_origin"].min(),
        "episode_end_origin": group["forecast_origin"].max(),
        "warning_months": warning_months,
        "detected_onsets": detected_onsets,
        "false_warning_months": false_warning_months,
        "mean_probability": group[
            "predicted_probability"
        ].mean(),
        "maximum_probability": group[
            "predicted_probability"
        ].max(),
        "episode_type": episode_type,
    })

episodes = pd.DataFrame(episode_records)

if not episodes.empty:
    episodes = episodes.sort_values(
        ["test_period", "episode_start_target"]
    ).reset_index(drop=True)

# ============================================================
# EVENT-LEVEL ONSET DETECTION
# ============================================================

onset_events = df[
    df["actual_onset"] == 1
].copy()

onset_events["event_detected"] = (
    onset_events["predicted_onset"] == 1
).astype(int)

onset_events["detection_status"] = np.where(
    onset_events["event_detected"] == 1,
    "detected",
    "missed"
)

onset_events["warning_episode_id"] = (
    onset_events["warning_episode_id"]
)

onset_events = onset_events[
    [
        "test_period",
        "target_date",
        "forecast_origin",
        "actual_onset",
        "predicted_probability",
        "predicted_onset",
        "event_detected",
        "detection_status",
        "warning_episode_id",
    ]
].sort_values("target_date")

# ============================================================
# OVERALL EPISODE METRICS
# ============================================================

total_observed_onsets = len(onset_events)

detected_onsets = int(
    onset_events["event_detected"].sum()
)

missed_onsets = (
    total_observed_onsets - detected_onsets
)

total_warning_months = len(warnings)

true_warning_months = int(
    warnings["actual_onset"].sum()
)

false_warning_months = (
    total_warning_months - true_warning_months
)

total_episodes = len(episodes)

if total_episodes:
    hit_only_episodes = int(
        (episodes["episode_type"] == "hit_only").sum()
    )

    mixed_episodes = int(
        (episodes["episode_type"] == "mixed").sum()
    )

    false_only_episodes = int(
        (episodes["episode_type"] == "false_only").sum()
    )

    episodes_with_onsets = (
        hit_only_episodes + mixed_episodes
    )

    episode_hit_proportion = (
        episodes_with_onsets / total_episodes
    )

    mean_episode_duration = (
        episodes["warning_months"].mean()
    )

    median_episode_duration = (
        episodes["warning_months"].median()
    )

    maximum_episode_duration = (
        episodes["warning_months"].max()
    )

else:
    hit_only_episodes = 0
    mixed_episodes = 0
    false_only_episodes = 0
    episodes_with_onsets = 0
    episode_hit_proportion = np.nan
    mean_episode_duration = np.nan
    median_episode_duration = np.nan
    maximum_episode_duration = 0

event_recall = (
    detected_onsets / total_observed_onsets
    if total_observed_onsets else np.nan
)

warning_month_precision = (
    true_warning_months / total_warning_months
    if total_warning_months else np.nan
)

summary = pd.DataFrame([{
    "model": MODEL,
    "eligible_test_months": len(df),
    "observed_onset_events": total_observed_onsets,
    "detected_onset_events": detected_onsets,
    "missed_onset_events": missed_onsets,
    "event_recall": event_recall,
    "total_warning_months": total_warning_months,
    "true_warning_months": true_warning_months,
    "false_warning_months": false_warning_months,
    "warning_month_precision": warning_month_precision,
    "total_warning_episodes": total_episodes,
    "hit_only_episodes": hit_only_episodes,
    "mixed_episodes": mixed_episodes,
    "false_only_episodes": false_only_episodes,
    "episodes_with_at_least_one_onset": episodes_with_onsets,
    "episode_hit_proportion": episode_hit_proportion,
    "mean_episode_duration_months": mean_episode_duration,
    "median_episode_duration_months": median_episode_duration,
    "maximum_episode_duration_months": maximum_episode_duration,
}])

# ============================================================
# SAVE OUTPUTS
# ============================================================

episodes.to_csv(
    OUTPUT_DIR / "warning_episode_details.csv",
    index=False
)

onset_events.to_csv(
    OUTPUT_DIR / "onset_event_detection_details.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "warning_episode_summary.csv",
    index=False
)

df.to_csv(
    OUTPUT_DIR / "classified_predictions_with_episode_ids.csv",
    index=False
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("EVENT-LEVEL DETECTION")
print("=" * 70)

print(f"Observed onset events: {total_observed_onsets}")
print(f"Detected onset events: {detected_onsets}")
print(f"Missed onset events: {missed_onsets}")
print(f"Event recall: {event_recall:.3f}")

print("\n" + "=" * 70)
print("WARNING MONTHS")
print("=" * 70)

print(f"Total warning months: {total_warning_months}")
print(f"True warning months: {true_warning_months}")
print(f"False warning months: {false_warning_months}")
print(f"Warning-month precision: {warning_month_precision:.3f}")

print("\n" + "=" * 70)
print("WARNING EPISODES")
print("=" * 70)

print(f"Total warning episodes: {total_episodes}")
print(f"Hit-only episodes: {hit_only_episodes}")
print(f"Mixed episodes: {mixed_episodes}")
print(f"False-only episodes: {false_only_episodes}")

print(
    "Episodes containing an onset: "
    f"{episodes_with_onsets}"
)

print(
    "Episode hit proportion: "
    f"{episode_hit_proportion:.3f}"
)

print(
    "Mean episode duration: "
    f"{mean_episode_duration:.2f} months"
)

print(
    "Median episode duration: "
    f"{median_episode_duration:.2f} months"
)

print(
    "Maximum episode duration: "
    f"{maximum_episode_duration} months"
)

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(OUTPUT_DIR.resolve())

for filename in [
    "warning_episode_summary.csv",
    "warning_episode_details.csv",
    "onset_event_detection_details.csv",
    "classified_predictions_with_episode_ids.csv",
]:
    print(f"Generated: {filename}")

