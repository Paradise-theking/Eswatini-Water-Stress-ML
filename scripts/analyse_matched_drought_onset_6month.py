from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT
    / "data"
    / "processed"
    / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)

OUTPUT_DIR = (
    PROJECT
    / "outputs"
    / "expanded"
    / "drought_onset_robustness_6month"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 10000
CONTROL_EXCLUSION_MONTHS = 6

LEAD_TIMES = [1, 2, 3]

PREDICTORS = [
    "precipitation_mm",
    "pet_mm",
    "water_balance_mm",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "water_balance_3month",
    "water_balance_6month",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_min_c",
    "dewpoint_c",
    "surface_solar_radiation_downwards_sum",
    "runoff_sum",
    "surface_runoff_sum",
]

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("=" * 70)
print("ROBUST DROUGHT-ONSET SIGNAL ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "month_date",
    "swba_3month",
] + PREDICTORS

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["month_date"] = pd.to_datetime(df["month_date"])

df = df.sort_values("month_date").copy()

df["period"] = df["month_date"].dt.to_period("M")

df = df.set_index("period")

full_periods = pd.period_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="M",
)

df = df.reindex(full_periods)

df.index.name = "period"

df["swba_3month"] = pd.to_numeric(
    df["swba_3month"],
    errors="coerce",
)

for col in PREDICTORS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

print(f"Dataset months: {len(df)}")
print(f"First month: {df.index.min()}")
print(f"Last month:  {df.index.max()}")


# ============================================================
# 3. CALCULATE SEASONAL ANOMALIES
#    Fixed reference climatology: 1981–2010
# ============================================================

reference = df.loc[
    (df.index.year >= 1981)
    & (df.index.year <= 2010)
]

for col in PREDICTORS:

    monthly_mean = (
        reference[col]
        .groupby(reference.index.month)
        .mean()
    )

    monthly_std = (
        reference[col]
        .groupby(reference.index.month)
        .std(ddof=0)
    )

    month_numbers = df.index.month

    expected_mean = pd.Series(
        month_numbers.map(monthly_mean),
        index=df.index,
    )

    expected_std = pd.Series(
        month_numbers.map(monthly_std),
        index=df.index,
    )

    df[f"anomaly_{col}"] = (
        df[col] - expected_mean
    ) / expected_std.replace(0, np.nan)


# ============================================================
# 4. CALCULATE ANTECEDENT CHANGES
# ============================================================

FEATURES = []

for col in PREDICTORS:

    FEATURES.append(f"anomaly_{col}")

    for lag in [1, 3, 6]:

        change_col = f"change_{lag}m_{col}"

        df[change_col] = (
            df[col] - df[col].shift(lag)
        )

        FEATURES.append(change_col)


# ============================================================
# 5. IDENTIFY DROUGHT MONTHS AND DROUGHT ONSETS
# ============================================================

df["is_drought"] = df["swba_3month"] <= -1

drought_periods = [
    period
    for period in df.index
    if bool(df.loc[period, "is_drought"])
]

drought_ordinals = np.array(
    [period.ordinal for period in drought_periods],
    dtype=int,
)


def close_to_drought(period, exclusion_months=3):

    if len(drought_ordinals) == 0:
        return False

    distance = np.abs(
        drought_ordinals - period.ordinal
    )

    return bool(
        np.any(distance <= exclusion_months)
    )


onset_periods = []

for period in df.index:

    current_swba = df.loc[period, "swba_3month"]

    previous_period = period - 1

    if previous_period not in df.index:
        continue

    previous_swba = df.loc[
        previous_period,
        "swba_3month",
    ]

    if (
        pd.notna(current_swba)
        and pd.notna(previous_swba)
        and current_swba <= -1
        and previous_swba > -1
    ):
        onset_periods.append(period)


print()
print(f"Historical drought months: {len(drought_periods)}")
print(f"Identified drought onsets: {len(onset_periods)}")


# ============================================================
# 6. CREATE MATCHED ONSET-CONTROL COMPARISONS
# ============================================================

event_records = []

for lead in LEAD_TIMES:

    print()
    print("-" * 70)
    print(f"ANALYSING {lead}-MONTH LEAD TIME")
    print("-" * 70)

    eligible_cases = 0
    matched_cases = 0

    for onset_period in onset_periods:

        origin_period = onset_period - lead

        if origin_period not in df.index:
            continue

        case_origin = df.loc[origin_period]

        case_origin_swba = case_origin["swba_3month"]

        # Only retain genuine advance-onset cases:
        # the catchment has not yet crossed the drought threshold.
        if (
            pd.isna(case_origin_swba)
            or case_origin_swba <= -1
        ):
            continue

        eligible_cases += 1

        # Candidate controls must have the same target
        # calendar month as this drought onset.
        candidate_targets = []

        for control_target in df.index:

            if control_target.month != onset_period.month:
                continue

            control_origin = control_target - lead

            if control_origin not in df.index:
                continue

            target_swba = df.loc[
                control_target,
                "swba_3month",
            ]

            origin_swba = df.loc[
                control_origin,
                "swba_3month",
            ]

            if (
                pd.isna(target_swba)
                or pd.isna(origin_swba)
            ):
                continue

            # Both control target and origin must be
            # non-drought months.
            if target_swba <= -1 or origin_swba <= -1:
                continue

            # Exclude controls close to any drought month.
            # Apply this to both target and forecast origin.
            if close_to_drought(
                control_target,
                CONTROL_EXCLUSION_MONTHS,
            ):
                continue

            if close_to_drought(
                control_origin,
                CONTROL_EXCLUSION_MONTHS,
            ):
                continue

            candidate_targets.append(control_target)

        if len(candidate_targets) == 0:
            continue

        matched_cases += 1

        # Extract the onset and matched-control values
        # at the forecast origin.
        for feature in FEATURES:

            case_value = case_origin[feature]

            if pd.isna(case_value):
                continue

            control_values = []

            for control_target in candidate_targets:

                control_origin = control_target - lead

                value = df.loc[
                    control_origin,
                    feature,
                ]

                if pd.notna(value):
                    control_values.append(float(value))

            if len(control_values) == 0:
                continue

            control_mean = float(
                np.mean(control_values)
            )

            event_records.append({
                "lead_months": lead,
                "onset_month": str(onset_period),
                "forecast_origin": str(origin_period),
                "target_calendar_month": onset_period.month,
                "feature": feature,
                "case_value": float(case_value),
                "matched_control_mean": control_mean,
                "paired_difference": (
                    float(case_value) - control_mean
                ),
                "number_of_controls": len(control_values),
            })

    print(f"Eligible onset cases: {eligible_cases}")
    print(f"Cases with matched controls: {matched_cases}")


# ============================================================
# 7. SAVE EVENT-LEVEL MATCHED DATA
# ============================================================

event_df = pd.DataFrame(event_records)

if event_df.empty:
    raise RuntimeError(
        "No matched comparisons were produced. "
        "Check the input data and eligibility rules."
    )

event_file = (
    OUTPUT_DIR
    / "matched_drought_onset_event_data.csv"
)

event_df.to_csv(event_file, index=False)

print()
print(f"Event-level data saved: {event_file}")


# ============================================================
# 8. BENJAMINI-HOCHBERG CORRECTION
# ============================================================

def benjamini_hochberg(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    n = len(p_values)

    order = np.argsort(p_values)

    ranked_p = p_values[order]

    adjusted = ranked_p * n / (
        np.arange(n) + 1
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(adjusted, 0, 1)

    result = np.empty(n)

    result[order] = adjusted

    return result


# ============================================================
# 9. PAIRED EFFECTS, WILCOXON TESTS AND EVENT BOOTSTRAP
# ============================================================

summary_records = []

for (lead, feature), group in event_df.groupby(
    ["lead_months", "feature"]
):

    differences = group[
        "paired_difference"
    ].dropna().to_numpy(dtype=float)

    case_values = group[
        "case_value"
    ].dropna().to_numpy(dtype=float)

    control_values = group[
        "matched_control_mean"
    ].dropna().to_numpy(dtype=float)

    n = len(differences)

    if n < 2:
        continue

    mean_difference = float(
        np.mean(differences)
    )

    median_difference = float(
        np.median(differences)
    )

    sd_difference = float(
        np.std(differences, ddof=1)
    )

    if sd_difference > 0:
        paired_effect_size = (
            mean_difference / sd_difference
        )
    else:
        paired_effect_size = np.nan

    # Paired Wilcoxon signed-rank test.
    if np.allclose(differences, 0):

        p_value = 1.0

    else:

        try:
            test_result = stats.wilcoxon(
                differences,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )

            p_value = float(test_result.pvalue)

        except ValueError:
            p_value = np.nan

    # Bootstrap by drought-onset event.
    # Each sampled unit is an onset and its matched contrast.
    bootstrap_means = []

    for _ in range(BOOTSTRAP_ITERATIONS):

        sampled = rng.choice(
            differences,
            size=n,
            replace=True,
        )

        bootstrap_means.append(
            np.mean(sampled)
        )

    ci_lower, ci_upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    summary_records.append({

        "lead_months": lead,
        "feature": feature,
        "n_matched_onsets": n,

        "mean_case_value": float(
            np.mean(case_values)
        ),

        "mean_matched_control": float(
            np.mean(control_values)
        ),

        "mean_paired_difference": mean_difference,

        "median_paired_difference": median_difference,

        "paired_effect_size": paired_effect_size,

        "bootstrap_95ci_lower": float(ci_lower),

        "bootstrap_95ci_upper": float(ci_upper),

        "wilcoxon_p_value": p_value,

        "mean_number_of_controls": float(
            group["number_of_controls"].mean()
        ),
    })


summary_df = pd.DataFrame(summary_records)

if summary_df.empty:
    raise RuntimeError(
        "No valid statistical comparisons were produced."
    )

# Correct across all tested features and lead times.
valid_p = summary_df["wilcoxon_p_value"].notna()

summary_df["bh_adjusted_p_value"] = np.nan

summary_df.loc[
    valid_p,
    "bh_adjusted_p_value",
] = benjamini_hochberg(
    summary_df.loc[
        valid_p,
        "wilcoxon_p_value",
    ].to_numpy()
)

summary_df["significant_after_bh"] = (
    summary_df["bh_adjusted_p_value"] < 0.05
)

summary_df["bootstrap_ci_excludes_zero"] = (
    (summary_df["bootstrap_95ci_lower"] > 0)
    | (summary_df["bootstrap_95ci_upper"] < 0)
)


# ============================================================
# 10. SAVE SUMMARY AND PRIORITY RANKING
# ============================================================

summary_df = summary_df.sort_values(
    [
        "lead_months",
        "bh_adjusted_p_value",
        "paired_effect_size",
    ],
    ascending=[True, True, False],
)

summary_file = (
    OUTPUT_DIR
    / "matched_drought_onset_statistical_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
)

priority_df = summary_df.copy()

priority_df["absolute_effect_size"] = (
    priority_df["paired_effect_size"].abs()
)

priority_df = priority_df.sort_values(
    [
        "bh_adjusted_p_value",
        "absolute_effect_size",
    ],
    ascending=[True, False],
)

priority_file = (
    OUTPUT_DIR
    / "matched_drought_onset_priority_ranking.csv"
)

priority_df.to_csv(
    priority_file,
    index=False,
)


# ============================================================
# 11. PRINT KEY RESULTS
# ============================================================

print()
print("=" * 70)
print("MATCHED DROUGHT-ONSET ANALYSIS COMPLETE")
print("=" * 70)

for lead in LEAD_TIMES:

    subset = summary_df[
        summary_df["lead_months"] == lead
    ].copy()

    if subset.empty:
        continue

    significant = subset[
        subset["significant_after_bh"]
    ]

    print()
    print(f"{lead}-MONTH LEAD")
    print("-" * 70)

    print(
        f"Features tested: {len(subset)}"
    )

    print(
        "Features significant after global BH correction: "
        f"{len(significant)}"
    )

    top = subset.sort_values(
        "bh_adjusted_p_value"
    ).head(8)

    display_columns = [
        "feature",
        "n_matched_onsets",
        "mean_paired_difference",
        "paired_effect_size",
        "bootstrap_95ci_lower",
        "bootstrap_95ci_upper",
        "bh_adjusted_p_value",
    ]

    print(
        top[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

print()
print("OUTPUT FILES")
print(summary_file)
print(priority_file)
print(event_file)

print()
print(
    "IMPORTANT: These are retrospective matched associations, "
    "not out-of-sample forecasting results."
)

