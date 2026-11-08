from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

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

DATE_COL = "month_date"
SWBA_COL = "swba_3month"

DROUGHT_THRESHOLD = -1.0
LEADS = [1, 2, 3]

FOLDS = [
    (2006, 2010),
    (2011, 2015),
    (2016, 2020),
    (2021, 2025),
]

# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

required_columns = [DATE_COL, SWBA_COL]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    errors="coerce"
)

df[SWBA_COL] = pd.to_numeric(
    df[SWBA_COL],
    errors="coerce"
)

df = (
    df.dropna(subset=[DATE_COL])
    .sort_values(DATE_COL)
    .reset_index(drop=True)
)

df["target_period"] = (
    df[DATE_COL].dt.to_period("M")
)

if df["target_period"].duplicated().any():
    duplicates = df.loc[
        df["target_period"].duplicated(keep=False),
        "target_period"
    ].tolist()

    raise ValueError(
        f"Duplicate monthly dates found: {duplicates[:10]}"
    )

# Retain the complete monthly calendar, including months
# where SWBA is unavailable, so previous-month comparisons
# cannot accidentally skip missing months.

df = df.set_index("target_period")

df.index.name = "target_period"

if not df.index.is_monotonic_increasing:
    raise ValueError("Monthly dates are not chronological.")

# Verify the expected date range.
print("=" * 75)
print("CHRONOLOGICAL DROUGHT-ONSET VALIDATION AUDIT")
print("=" * 75)

print("\nDataset period:")
print(f"  Start: {df.index.min()}")
print(f"  End:   {df.index.max()}")
print(f"  Rows:  {len(df)}")

if len(df) != 540:
    print(
        "\nWARNING: Expected 540 monthly rows. "
        "Please verify the dataset."
    )

# ============================================================
# CREATE SWBA LOOKUP AND DROUGHT LABELS
# ============================================================

swba_lookup = df[SWBA_COL].to_dict()

df["previous_period"] = (
    df.index - 1
)

df["previous_swba"] = (
    df["previous_period"].map(swba_lookup)
)

df["is_drought"] = (
    df[SWBA_COL].notna()
    & (df[SWBA_COL] <= DROUGHT_THRESHOLD)
)

df["drought_onset"] = (
    df["is_drought"]
    & df["previous_swba"].notna()
    & (df["previous_swba"] > DROUGHT_THRESHOLD)
)

# Identify continuous drought episodes.
previous_is_drought = (
    df["is_drought"].shift(1, fill_value=False)
)

df["new_episode"] = (
    df["is_drought"] & ~previous_is_drought
)

df["episode_id"] = (
    df["new_episode"].cumsum()
)

df.loc[
    ~df["is_drought"],
    "episode_id"
] = np.nan

valid_swba = df[SWBA_COL].dropna()

print("\nHistorical drought summary:")
print(f"  Valid SWBA observations: {len(valid_swba)}")
print(f"  Drought months: {int(df['is_drought'].sum())}")
print(f"  Drought onsets: {int(df['drought_onset'].sum())}")
print(f"  Drought episodes: {int(df['episode_id'].nunique())}")

# ============================================================
# BUILD LEAD-SPECIFIC CLASSIFICATION SAMPLES
# ============================================================

all_results = []

for lead in LEADS:

    print("\n" + "=" * 75)
    print(f"{lead}-MONTH FORECAST LEAD")
    print("=" * 75)

    data = df[
        [SWBA_COL, "previous_swba", "is_drought",
         "drought_onset", "episode_id"]
    ].copy()

    data = data.dropna(subset=[SWBA_COL])

    data["forecast_origin"] = (
        data.index - lead
    )

    # SWBA available at the forecast origin.
    data["origin_swba"] = (
        data["forecast_origin"].map(swba_lookup)
    )

    # Only retain genuine advance-warning opportunities:
    # the catchment must not already be in drought at origin.
    data = data.dropna(
        subset=["origin_swba"]
    ).copy()

    data = data[
        data["origin_swba"] > DROUGHT_THRESHOLD
    ].copy()

    data["onset_label"] = (
        data["drought_onset"].astype(int)
    )

    print(f"Eligible observations: {len(data)}")
    print(
        f"Eligible onset cases: "
        f"{int(data['onset_label'].sum())}"
    )
    print(
        f"Eligible non-onset cases: "
        f"{int((data['onset_label'] == 0).sum())}"
    )

    # ========================================================
    # CHRONOLOGICAL FOLDS
    # ========================================================

    for test_start, test_end in FOLDS:

        first_test_target = pd.Period(
            f"{test_start}-01",
            freq="M"
        )

        first_test_origin = (
            first_test_target - lead
        )

        # Only training labels observable by the first
        # forecast origin in the test period are allowed.
        train = data[
            data.index <= first_test_origin
        ].copy()

        test = data[
            (data.index.year >= test_start)
            & (data.index.year <= test_end)
        ].copy()

        train_onsets = int(
            train["onset_label"].sum()
        )

        train_non_onsets = int(
            (train["onset_label"] == 0).sum()
        )

        test_onsets = int(
            test["onset_label"].sum()
        )

        test_non_onsets = int(
            (test["onset_label"] == 0).sum()
        )

        # All drought episodes present in the test calendar
        # period, regardless of whether their onset was
        # eligible for advance prediction at this lead.
        calendar_test = df[
            (df.index.year >= test_start)
            & (df.index.year <= test_end)
        ]

        test_drought_months = calendar_test[
            calendar_test["is_drought"]
        ]

        episodes_present = int(
            test_drought_months["episode_id"].nunique()
        )

        calendar_onsets = int(
            calendar_test["drought_onset"].sum()
        )

        if test_onsets == 0:
            assessment = "NO ELIGIBLE POSITIVE TEST CASES"
        elif test_onsets < 5:
            assessment = "VERY FEW POSITIVE TEST CASES"
        elif test_onsets < 10:
            assessment = "LIMITED POSITIVE TEST CASES"
        else:
            assessment = "REVIEW EVENT COUNT AND INDEPENDENCE"

        result = {
            "lead_months": lead,
            "test_period": f"{test_start}-{test_end}",
            "first_test_origin": str(first_test_origin),
            "training_observations": len(train),
            "training_onsets": train_onsets,
            "training_non_onsets": train_non_onsets,
            "test_observations": len(test),
            "test_onsets": test_onsets,
            "test_non_onsets": test_non_onsets,
            "calendar_onsets": calendar_onsets,
            "drought_episodes_present": episodes_present,
            "assessment": assessment,
        }

        all_results.append(result)

        print(f"\nTest period: {test_start}-{test_end}")
        print(f"  First test forecast origin: {first_test_origin}")
        print(f"  Training observations: {len(train)}")
        print(f"  Training onset cases: {train_onsets}")
        print(f"  Training non-onset cases: {train_non_onsets}")
        print(f"  Test observations: {len(test)}")
        print(f"  Eligible test onset cases: {test_onsets}")
        print(f"  Eligible test non-onset cases: {test_non_onsets}")
        print(f"  Calendar-period onset events: {calendar_onsets}")
        print(f"  Drought episodes present: {episodes_present}")
        print(f"  Assessment: {assessment}")

# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(all_results)

output_file = (
    OUTPUT_DIR
    / "chronological_onset_validation_audit.csv"
)

results.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 75)
print("VALIDATION AUDIT SUMMARY")
print("=" * 75)

summary_columns = [
    "lead_months",
    "test_period",
    "training_observations",
    "training_onsets",
    "test_observations",
    "test_onsets",
    "drought_episodes_present",
    "assessment",
]

print(
    results[summary_columns].to_string(index=False)
)

print("\nSaved results:")
print(output_file)

print("\nNo classification models have been trained.")
