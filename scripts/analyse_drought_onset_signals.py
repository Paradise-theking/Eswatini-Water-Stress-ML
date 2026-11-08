from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# ============================================================
# DROUGHT ONSET SIGNAL AND LEAD-TIME ANALYSIS
# Mnjoli W60E Catchment
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "expanded"
    / "drought_onset_signal_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_START = "1981-01-01"
REFERENCE_END = "2010-12-31"

DROUGHT_THRESHOLD = -1.0
LEAD_MONTHS = [1, 2, 3]
CHANGE_LAGS = [1, 3, 6]

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

# SWBA is retained as a separate antecedent-condition indicator.
INDEX_VARIABLE = "swba_3month"

print("=" * 76)
print("DROUGHT ONSET SIGNAL AND LEAD-TIME ANALYSIS")
print("Mnjoli W60E Catchment")
print("=" * 76)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

data = pd.read_csv(INPUT_FILE)

data["month_date"] = pd.to_datetime(data["month_date"])

data = (
    data.sort_values("month_date")
    .drop_duplicates(subset="month_date")
    .set_index("month_date")
)

required = PREDICTORS + [INDEX_VARIABLE]

missing = [column for column in required if column not in data.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

for column in required:
    data[column] = pd.to_numeric(data[column], errors="coerce")

print(f"\nMonthly records: {len(data)}")
print(f"Data period: {data.index.min():%Y-%m} to {data.index.max():%Y-%m}")
print(f"Predictors: {len(PREDICTORS)}")

# ------------------------------------------------------------
# 2. CALCULATE FIXED SEASONAL CLIMATOLOGY
# ------------------------------------------------------------

reference = data.loc[REFERENCE_START:REFERENCE_END].copy()

if reference.empty:
    raise ValueError("The 1981-2010 reference period is unavailable.")

climatology = (
    reference.groupby(reference.index.month)[PREDICTORS]
    .agg(["mean", "std"])
)

# Calculate calendar-month standardised anomalies.
# The fixed reference period is used consistently for all years.
for predictor in PREDICTORS:

    monthly_mean = data.index.month.map(
        climatology[predictor]["mean"]
    )

    monthly_std = data.index.month.map(
        climatology[predictor]["std"]
    )

    monthly_mean = np.asarray(monthly_mean, dtype=float)
    monthly_std = np.asarray(monthly_std, dtype=float)

    values = data[predictor].to_numpy(dtype=float)

    anomaly = np.full(len(data), np.nan)

    valid = (
        np.isfinite(values)
        & np.isfinite(monthly_mean)
        & np.isfinite(monthly_std)
        & (monthly_std > 0)
    )

    anomaly[valid] = (
        values[valid] - monthly_mean[valid]
    ) / monthly_std[valid]

    data[f"z_{predictor}"] = anomaly

print("\nSeasonal anomalies calculated using 1981-2010 climatology.")

# ------------------------------------------------------------
# 3. IDENTIFY DROUGHT ONSETS
# ------------------------------------------------------------

onset_dates = []

for i in range(1, len(data)):

    current_date = data.index[i]
    previous_date = data.index[i - 1]

    # Ensure consecutive monthly observations.
    if current_date != previous_date + pd.DateOffset(months=1):
        continue

    current_swba = data.iloc[i][INDEX_VARIABLE]
    previous_swba = data.iloc[i - 1][INDEX_VARIABLE]

    if pd.isna(current_swba) or pd.isna(previous_swba):
        continue

    if (
        current_swba <= DROUGHT_THRESHOLD
        and previous_swba > DROUGHT_THRESHOLD
    ):
        onset_dates.append(current_date)

print(f"\nDrought onsets identified: {len(onset_dates)}")

if len(onset_dates) != 30:
    raise ValueError(
        f"Expected 30 onsets from the previous audit, "
        f"but identified {len(onset_dates)}. "
        "Stop and verify the source data before interpreting results."
    )

# ------------------------------------------------------------
# 4. BUILD EVENT AND CONTROL OBSERVATIONS
# ------------------------------------------------------------

records = []

for lead in LEAD_MONTHS:

    # Each drought onset is evaluated at this forecast lead.
    for onset_date in onset_dates:

        origin_date = onset_date - pd.DateOffset(months=lead)

        if origin_date not in data.index:
            continue

        origin = data.loc[origin_date]
        target = data.loc[onset_date]

        if pd.isna(origin[INDEX_VARIABLE]):
            continue

        if pd.isna(target[INDEX_VARIABLE]):
            continue

        # Forecast origin must still be outside drought.
        if origin[INDEX_VARIABLE] <= DROUGHT_THRESHOLD:
            continue

        record = {
            "lead_months": lead,
            "group": "Drought onset",
            "target_month": onset_date,
            "forecast_origin_month": origin_date,
            "calendar_month": onset_date.month,
            "origin_swba": origin[INDEX_VARIABLE],
            "target_swba": target[INDEX_VARIABLE],
        }

        for predictor in PREDICTORS:

            record[f"origin_{predictor}"] = origin[predictor]
            record[f"anomaly_{predictor}"] = origin[
                f"z_{predictor}"
            ]

            for lag in CHANGE_LAGS:

                previous_date = (
                    origin_date - pd.DateOffset(months=lag)
                )

                if previous_date in data.index:
                    previous_value = data.loc[
                        previous_date, predictor
                    ]
                else:
                    previous_value = np.nan

                current_value = origin[predictor]

                if (
                    pd.notna(current_value)
                    and pd.notna(previous_value)
                ):
                    change = current_value - previous_value
                else:
                    change = np.nan

                record[
                    f"change_{lag}m_{predictor}"
                ] = change

        records.append(record)

    # --------------------------------------------------------
    # Seasonally matched non-drought controls
    # --------------------------------------------------------

    onset_calendar_months = {
        date.month for date in onset_dates
    }

    for target_date in data.index:

        if target_date.month not in onset_calendar_months:
            continue

        target = data.loc[target_date]

        if pd.isna(target[INDEX_VARIABLE]):
            continue

        # Controls must not be drought months.
        if target[INDEX_VARIABLE] <= DROUGHT_THRESHOLD:
            continue

        origin_date = (
            target_date - pd.DateOffset(months=lead)
        )

        if origin_date not in data.index:
            continue

        origin = data.loc[origin_date]

        if pd.isna(origin[INDEX_VARIABLE]):
            continue

        # Exclude controls whose forecast origin is already in drought.
        if origin[INDEX_VARIABLE] <= DROUGHT_THRESHOLD:
            continue

        record = {
            "lead_months": lead,
            "group": "Non-drought control",
            "target_month": target_date,
            "forecast_origin_month": origin_date,
            "calendar_month": target_date.month,
            "origin_swba": origin[INDEX_VARIABLE],
            "target_swba": target[INDEX_VARIABLE],
        }

        for predictor in PREDICTORS:

            record[f"origin_{predictor}"] = origin[predictor]
            record[f"anomaly_{predictor}"] = origin[
                f"z_{predictor}"
            ]

            for lag in CHANGE_LAGS:

                previous_date = (
                    origin_date - pd.DateOffset(months=lag)
                )

                if previous_date in data.index:
                    previous_value = data.loc[
                        previous_date, predictor
                    ]
                else:
                    previous_value = np.nan

                current_value = origin[predictor]

                if (
                    pd.notna(current_value)
                    and pd.notna(previous_value)
                ):
                    change = current_value - previous_value
                else:
                    change = np.nan

                record[
                    f"change_{lag}m_{predictor}"
                ] = change

        records.append(record)

pairs = pd.DataFrame(records)

# ------------------------------------------------------------
# 5. SAVE THE EVENT-CONTROL DATASET
# ------------------------------------------------------------

pairs_file = (
    OUTPUT_DIR / "drought_onset_signal_comparison_data.csv"
)

pairs.to_csv(pairs_file, index=False)

# ------------------------------------------------------------
# 6. COMPARE SIGNALS AT EACH LEAD TIME
# ------------------------------------------------------------

features = []

for predictor in PREDICTORS:

    features.append(f"origin_{predictor}")
    features.append(f"anomaly_{predictor}")

    for lag in CHANGE_LAGS:
        features.append(f"change_{lag}m_{predictor}")

results = []

for lead in LEAD_MONTHS:

    lead_data = pairs[pairs["lead_months"] == lead]

    onset_data = lead_data[
        lead_data["group"] == "Drought onset"
    ]

    control_data = lead_data[
        lead_data["group"] == "Non-drought control"
    ]

    print("\n" + "-" * 76)
    print(f"LEAD TIME: {lead} MONTH(S)")
    print("-" * 76)

    print(f"Usable drought-onset cases: {len(onset_data)}")
    print(f"Non-drought control cases: {len(control_data)}")

    for feature in features:

        onset_values = onset_data[feature].dropna()
        control_values = control_data[feature].dropna()

        if len(onset_values) < 3 or len(control_values) < 3:
            continue

        onset_mean = onset_values.mean()
        control_mean = control_values.mean()

        onset_median = onset_values.median()
        control_median = control_values.median()

        onset_sd = onset_values.std(ddof=1)
        control_sd = control_values.std(ddof=1)

        pooled_sd = np.sqrt(
            (onset_sd ** 2 + control_sd ** 2) / 2
        )

        if pooled_sd > 0:
            smd = (onset_mean - control_mean) / pooled_sd
        else:
            smd = np.nan

        test = mannwhitneyu(
            onset_values,
            control_values,
            alternative="two-sided",
            method="auto",
        )

        results.append({
            "lead_months": lead,
            "feature": feature,
            "onset_n": len(onset_values),
            "control_n": len(control_values),
            "onset_mean": onset_mean,
            "control_mean": control_mean,
            "mean_difference": onset_mean - control_mean,
            "onset_median": onset_median,
            "control_median": control_median,
            "median_difference": onset_median - control_median,
            "standardized_mean_difference": smd,
            "absolute_smd": abs(smd) if pd.notna(smd) else np.nan,
            "mann_whitney_p_value": test.pvalue,
        })

summary = pd.DataFrame(results)

if summary.empty:
    raise ValueError("No valid event-control comparisons were produced.")

# ------------------------------------------------------------
# 7. APPLY BENJAMINI-HOCHBERG MULTIPLE-TEST CORRECTION
# ------------------------------------------------------------

summary["bh_adjusted_p_value"] = np.nan

for lead in LEAD_MONTHS:

    mask = summary["lead_months"] == lead
    p_values = summary.loc[
        mask, "mann_whitney_p_value"
    ].to_numpy(dtype=float)

    valid = np.isfinite(p_values)

    if valid.sum() == 0:
        continue

    valid_p = p_values[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]

    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)

    # Enforce monotonicity from largest rank backwards.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    restored = np.empty(m)
    restored[order] = adjusted

    original_indices = summary.index[mask][valid]
    summary.loc[
        original_indices, "bh_adjusted_p_value"
    ] = restored

# ------------------------------------------------------------
# 8. SAVE SUMMARY AND LEAD-TIME RANKING
# ------------------------------------------------------------

summary = summary.sort_values(
    ["lead_months", "absolute_smd"],
    ascending=[True, False]
)

summary_file = (
    OUTPUT_DIR / "drought_onset_signal_comparison_summary.csv"
)

summary.to_csv(summary_file, index=False)

# Compact ranking of seasonal anomalies and depletion rates.
priority_features = summary[
    summary["feature"].str.startswith(
        ("anomaly_", "change_1m_", "change_3m_", "change_6m_")
    )
].copy()

priority_file = (
    OUTPUT_DIR / "drought_onset_signal_priority_ranking.csv"
)

priority_features.to_csv(priority_file, index=False)

# ------------------------------------------------------------
# 9. PRINT RESULTS
# ------------------------------------------------------------

for lead in LEAD_MONTHS:

    print("\n" + "=" * 76)
    print(f"TOP SIGNALS AT {lead}-MONTH LEAD")
    print("=" * 76)

    selected = priority_features[
        priority_features["lead_months"] == lead
    ].head(15)

    columns = [
        "feature",
        "onset_n",
        "control_n",
        "mean_difference",
        "standardized_mean_difference",
        "mann_whitney_p_value",
        "bh_adjusted_p_value",
    ]

    print(
        selected[columns]
        .round(4)
        .to_string(index=False)
    )

# ------------------------------------------------------------
# 10. REPORT EVENT COVERAGE
# ------------------------------------------------------------

print("\n" + "=" * 76)
print("EVENT COVERAGE BY LEAD TIME")
print("=" * 76)

for lead in LEAD_MONTHS:

    count = pairs[
        (pairs["lead_months"] == lead)
        & (pairs["group"] == "Drought onset")
    ]["target_month"].nunique()

    print(f"{lead}-month lead: {count} of {len(onset_dates)} onsets")

print("\n" + "=" * 76)
print("ANALYSIS COMPLETED")
print("=" * 76)

print(f"\nComparison data:\n{pairs_file}")
print(f"\nFull statistical summary:\n{summary_file}")
print(f"\nPriority ranking:\n{priority_file}")

print(
    "\nSCIENTIFIC CAUTION:\n"
    "- The climatology is a fixed 1981-2010 reference.\n"
    "- Controls are matched by target calendar month, not individually paired.\n"
    "- Mann-Whitney tests are exploratory and do not establish causation.\n"
    "- Multiple-testing adjusted p-values are provided.\n"
    "- Historical association does not establish out-of-sample warning skill.\n"
    "- A separate chronological hindcast is required before claiming "
    "predictive performance."
)
