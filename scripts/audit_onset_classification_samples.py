from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

INPUT = (
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
    / "drought_onset_model_preparation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

df["month_date"] = pd.to_datetime(df["month_date"])
df = df.sort_values("month_date").copy()

df["period"] = df["month_date"].dt.to_period("M")
df = df.set_index("period")

df["swba_3month"] = pd.to_numeric(
    df["swba_3month"],
    errors="coerce",
)

df["previous_swba"] = df["swba_3month"].shift(1)

df["drought_onset"] = (
    (df["swba_3month"] <= -1)
    & (df["previous_swba"] > -1)
)

df["decade_group"] = (
    (df.index.year // 10) * 10
).astype(int)

records = []

for lead in [1, 2, 3]:

    for target_period in df.index:

        origin_period = target_period - lead

        if origin_period not in df.index:
            continue

        origin_swba = df.loc[
            origin_period,
            "swba_3month",
        ]

        target_swba = df.loc[
            target_period,
            "swba_3month",
        ]

        if pd.isna(origin_swba) or pd.isna(target_swba):
            continue

        # Only assess genuine advance-warning opportunities.
        if origin_swba <= -1:
            continue

        records.append({
            "lead_months": lead,
            "forecast_origin": str(origin_period),
            "target_month": str(target_period),
            "target_year": target_period.year,
            "decade_group": (
                target_period.year // 10
            ) * 10,
            "origin_swba": origin_swba,
            "target_swba": target_swba,
            "onset": int(
                bool(df.loc[target_period, "drought_onset"])
            ),
        })

samples = pd.DataFrame(records)

samples.to_csv(
    OUTPUT_DIR / "onset_classification_sample_inventory.csv",
    index=False,
)

print("=" * 75)
print("DROUGHT-ONSET CLASSIFICATION SAMPLE AUDIT")
print("=" * 75)

for lead in [1, 2, 3]:

    subset = samples[
        samples["lead_months"] == lead
    ]

    print()
    print(f"{lead}-MONTH LEAD")
    print("-" * 75)

    print(f"Total eligible observations: {len(subset)}")
    print(f"Onset cases: {subset['onset'].sum()}")
    print(f"Non-onset cases: {(subset['onset'] == 0).sum()}")

    print()
    print("Distribution by decade:")

    decade_summary = (
        subset.groupby("decade_group")
        .agg(
            observations=("onset", "size"),
            onset_cases=("onset", "sum"),
        )
    )

    decade_summary["non_onset_cases"] = (
        decade_summary["observations"]
        - decade_summary["onset_cases"]
    )

    print(decade_summary.to_string())

print()
print("Saved sample inventory:")
print(
    OUTPUT_DIR
    / "onset_classification_sample_inventory.csv"
)
