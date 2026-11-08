from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "expanded"
    / "drought_onset_model_preparation"
    / "data_audit"
)

DATE_COLUMN = "month_date"
SWBA_COLUMN = "swba_3month"
THRESHOLD = -1.0
BASELINE_START = 1981
BASELINE_END = 2010
LEADS = (1, 2, 3)
TEST_PERIODS = ((2006, 2010), (2011, 2015), (2016, 2020), (2021, 2025))
HYDRO_FEATURES = (
    "precipitation_mm",
    "pet_mm",
    "volumetric_soil_water_layer_1",
    "water_balance_3month",
)
FLOAT_TOLERANCE = 1e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison_summary(
    observed: pd.Series,
    reconstructed: pd.Series,
) -> dict[str, Any]:
    observed_values = pd.to_numeric(observed, errors="coerce").to_numpy(
        dtype=float
    )
    reconstructed_values = pd.to_numeric(
        reconstructed, errors="coerce"
    ).to_numpy(dtype=float)

    if len(observed_values) != len(reconstructed_values):
        raise ValueError("Cannot compare series with different lengths.")

    both_missing = np.isnan(observed_values) & np.isnan(reconstructed_values)
    finite_pairs = np.isfinite(observed_values) & np.isfinite(
        reconstructed_values
    )
    mismatched_missing = (
        np.isnan(observed_values) ^ np.isnan(reconstructed_values)
    )
    differences = np.abs(
        observed_values[finite_pairs] - reconstructed_values[finite_pairs]
    )

    return {
        "compared_rows": int(finite_pairs.sum()),
        "both_missing_rows": int(both_missing.sum()),
        "mismatched_missing_rows": int(mismatched_missing.sum()),
        "maximum_absolute_difference": (
            float(differences.max()) if differences.size else None
        ),
        "mean_absolute_difference": (
            float(differences.mean()) if differences.size else None
        ),
        "within_tolerance": bool(
            not mismatched_missing.any()
            and differences.size > 0
            and differences.max() <= FLOAT_TOLERANCE
        ),
    }


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "detail": detail,
        }
    )


def load_data(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    data = pd.read_csv(input_path)
    required = [
        DATE_COLUMN,
        "year",
        "month",
        "precipitation_mm",
        "pet_mm",
        "water_balance_mm",
        "water_balance_3month",
        "climatology_mean",
        "climatology_std",
        SWBA_COLUMN,
        *HYDRO_FEATURES,
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    if data[DATE_COLUMN].isna().any():
        raise ValueError("Dataset contains invalid or missing monthly dates.")

    data = data.sort_values(DATE_COLUMN).reset_index(drop=True)
    periods = data[DATE_COLUMN].dt.to_period("M")
    if periods.duplicated().any():
        duplicate_periods = periods[periods.duplicated()].astype(str).tolist()
        raise ValueError(f"Duplicate monthly periods found: {duplicate_periods}")

    expected_periods = pd.period_range(
        periods.min(),
        periods.max(),
        freq="M",
    )
    missing_periods = expected_periods.difference(
        pd.PeriodIndex(periods, freq="M")
    )
    if len(missing_periods):
        raise ValueError(
            "Missing monthly periods found: "
            + ", ".join(str(period) for period in missing_periods)
        )

    data["period"] = periods
    data["calendar_month"] = data[DATE_COLUMN].dt.month
    data["calendar_year"] = data[DATE_COLUMN].dt.year
    return data


def reconstruct_index(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["reconstructed_water_balance_mm"] = (
        result["precipitation_mm"] - result["pet_mm"]
    )
    result["reconstructed_water_balance_3month"] = (
        result["reconstructed_water_balance_mm"]
        .rolling(window=3, min_periods=3)
        .sum()
    )

    baseline = result.loc[
        result["calendar_year"].between(BASELINE_START, BASELINE_END)
    ]
    climatology = baseline.groupby("calendar_month")[
        "reconstructed_water_balance_3month"
    ].agg(
        reconstructed_climatology_mean="mean",
        reconstructed_climatology_std=lambda values: values.std(ddof=1),
    )
    result = result.join(climatology, on="calendar_month")
    result["reconstructed_swba"] = (
        result["reconstructed_water_balance_3month"]
        - result["reconstructed_climatology_mean"]
    ) / result["reconstructed_climatology_std"]
    return result


def label_onsets(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = data.copy()
    swba_by_period = result.set_index("period")[SWBA_COLUMN]
    previous_period = result["period"] - 1
    result["previous_month_swba"] = previous_period.map(swba_by_period)
    valid_transition = (
        result[SWBA_COLUMN].notna()
        & result["previous_month_swba"].notna()
    )
    result["onset_label"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )
    result.loc[valid_transition, "onset_label"] = (
        (
            result.loc[valid_transition, SWBA_COLUMN] <= THRESHOLD
        )
        & (
            result.loc[valid_transition, "previous_month_swba"] > THRESHOLD
        )
    ).astype(int)

    events = result.loc[
        result["onset_label"].eq(1),
        [
            DATE_COLUMN,
            "period",
            "previous_month_swba",
            SWBA_COLUMN,
        ],
    ].rename(
        columns={
            DATE_COLUMN: "onset_month",
            "period": "onset_period",
            "previous_month_swba": "previous_month_swba",
            SWBA_COLUMN: "onset_swba",
        }
    )
    events["forecast_origin_1_month"] = (
        events["onset_month"] - pd.DateOffset(months=1)
    )
    events["forecast_origin_2_months"] = (
        events["onset_month"] - pd.DateOffset(months=2)
    )
    events["forecast_origin_3_months"] = (
        events["onset_month"] - pd.DateOffset(months=3)
    )
    return result, events


def build_sample_and_fold_audits(
    labeled: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    onset_by_period = labeled.set_index("period")["onset_label"]
    swba_by_period = labeled.set_index("period")[SWBA_COLUMN]
    features_by_period = labeled.set_index("period")[list(HYDRO_FEATURES)]

    fold_rows: list[dict[str, Any]] = []
    sample_rows: list[pd.DataFrame] = []

    for lead in LEADS:
        samples = labeled.loc[
            labeled["onset_label"].notna(),
            ["period", DATE_COLUMN, "onset_label"],
        ].copy()
        samples["forecast_origin_period"] = samples["period"] - lead
        samples["origin_swba"] = samples[
            "forecast_origin_period"
        ].map(swba_by_period)
        samples = samples.loc[
            samples["origin_swba"].notna()
            & (samples["origin_swba"] > THRESHOLD)
        ].copy()
        samples["lead_months"] = lead
        for feature in HYDRO_FEATURES:
            samples[feature] = samples[
                "forecast_origin_period"
            ].map(features_by_period[feature])
        samples["predictors_complete"] = samples[
            list(HYDRO_FEATURES)
        ].notna().all(axis=1)
        sample_rows.append(samples)

        for test_start, test_end in TEST_PERIODS:
            validation_start = test_start - 5
            validation_end = test_start - 1
            validation_cutoff = (
                pd.Period(f"{validation_start}-01", freq="M") - lead
            )
            test_cutoff = pd.Period(f"{test_start}-01", freq="M") - lead

            training = samples.loc[
                (samples["period"] <= validation_cutoff)
                & samples["predictors_complete"]
            ]
            validation = samples.loc[
                samples["period"].dt.year.between(
                    validation_start, validation_end
                )
                & samples["predictors_complete"]
            ]
            final_training = samples.loc[
                (samples["period"] <= test_cutoff)
                & samples["predictors_complete"]
            ]
            test = samples.loc[
                samples["period"].dt.year.between(test_start, test_end)
                & samples["predictors_complete"]
            ]

            fold_rows.append(
                {
                    "lead_months": lead,
                    "test_period": f"{test_start}-{test_end}",
                    "validation_period": (
                        f"{validation_start}-{validation_end}"
                    ),
                    "validation_training_cutoff": str(validation_cutoff),
                    "test_training_cutoff": str(test_cutoff),
                    "validation_training_samples": len(training),
                    "validation_training_onsets": int(
                        training["onset_label"].sum()
                    ),
                    "validation_samples": len(validation),
                    "validation_onsets": int(
                        validation["onset_label"].sum()
                    ),
                    "final_training_samples": len(final_training),
                    "final_training_onsets": int(
                        final_training["onset_label"].sum()
                    ),
                    "test_samples": len(test),
                    "test_onsets": int(test["onset_label"].sum()),
                    "test_non_onsets": int(
                        len(test) - test["onset_label"].sum()
                    ),
                }
            )

    return pd.concat(sample_rows, ignore_index=True), pd.DataFrame(fold_rows)


def audit(
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    data = load_data(input_path)
    reconstructed = reconstruct_index(data)
    labeled, events = label_onsets(reconstructed)
    samples, folds = build_sample_and_fold_audits(labeled)

    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "date_components_match",
        bool(
            (data["year"].astype(int) == data["calendar_year"]).all()
            and (data["month"].astype(int) == data["calendar_month"]).all()
        ),
        "Stored year and month columns match month_date.",
    )

    comparisons = {
        "monthly_water_balance_matches_precipitation_minus_pet": (
            data["water_balance_mm"],
            reconstructed["reconstructed_water_balance_mm"],
        ),
        "three_month_water_balance_matches_rolling_sum": (
            data["water_balance_3month"],
            reconstructed["reconstructed_water_balance_3month"],
        ),
        "stored_climatology_mean_matches_1981_2010": (
            data["climatology_mean"],
            reconstructed["reconstructed_climatology_mean"],
        ),
        "stored_climatology_std_matches_1981_2010_sample_std": (
            data["climatology_std"],
            reconstructed["reconstructed_climatology_std"],
        ),
        "stored_swba_matches_reconstructed_swba": (
            data[SWBA_COLUMN],
            reconstructed["reconstructed_swba"],
        ),
    }

    comparison_results: dict[str, Any] = {}
    for name, (observed, expected) in comparisons.items():
        result = comparison_summary(observed, expected)
        comparison_results[name] = result
        add_check(
            checks,
            name,
            result["within_tolerance"],
            (
                f"Compared {result['compared_rows']} finite rows; "
                f"maximum absolute difference="
                f"{result['maximum_absolute_difference']}; "
                f"tolerance={FLOAT_TOLERANCE}."
            ),
        )

    missingness = {
        column: int(data[column].isna().sum())
        for column in [SWBA_COLUMN, *HYDRO_FEATURES]
    }
    add_check(
        checks,
        "monthly_periods_are_contiguous",
        True,
        (
            f"{len(data)} unique, contiguous months from "
            f"{data[DATE_COLUMN].min():%Y-%m} to "
            f"{data[DATE_COLUMN].max():%Y-%m}."
        ),
    )
    add_check(
        checks,
        "eligible_forecast_samples_have_complete_predictors",
        bool(samples["predictors_complete"].all()),
        (
            f"{int((~samples['predictors_complete']).sum())} of "
            f"{len(samples)} eligible lead-specific samples have missing "
            "candidate predictors. Initial rolling-window warm-up rows are "
            f"reported separately: {missingness}."
        ),
    )

    valid_labels = labeled["onset_label"].dropna().astype(int)
    onset_counts_by_year = (
        labeled.loc[labeled["onset_label"].eq(1)]
        .groupby("calendar_year")
        .size()
        .astype(int)
        .to_dict()
    )

    report: dict[str, Any] = {
        "input_file": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "audit_script": str(Path(__file__).resolve()),
        "definition": {
            "index": "Three-month precipitation-minus-PET water-balance anomaly.",
            "climatology_period": f"{BASELINE_START}-{BASELINE_END}",
            "climatology_standard_deviation": "Sample standard deviation (ddof=1).",
            "drought_threshold": THRESHOLD,
            "onset_label": (
                f"SWBA <= {THRESHOLD} in the target month, with the "
                "immediately preceding calendar month SWBA > "
                f"{THRESHOLD}."
            ),
            "forecast_origin": (
                "Only samples with non-drought SWBA at the lead-specific "
                "forecast origin are eligible."
            ),
        },
        "dataset": {
            "rows": len(data),
            "start_month": data[DATE_COLUMN].min().strftime("%Y-%m"),
            "end_month": data[DATE_COLUMN].max().strftime("%Y-%m"),
            "missing_values": missingness,
        },
        "reconstruction_comparisons": comparison_results,
        "onsets": {
            "valid_adjacent_month_transitions": int(valid_labels.size),
            "drought_months": int(
                (labeled[SWBA_COLUMN] <= THRESHOLD).sum()
            ),
            "onset_events": int(len(events)),
            "onsets_by_year": {
                str(year): count
                for year, count in onset_counts_by_year.items()
            },
        },
        "eligible_forecast_samples": {
            "rows_by_lead": {
                str(lead): int((samples["lead_months"] == lead).sum())
                for lead in LEADS
            },
            "incomplete_predictor_rows": int(
                (~samples["predictors_complete"]).sum()
            ),
        },
        "folds": folds.to_dict(orient="records"),
        "checks": checks,
        "all_checks_passed": all(check["passed"] for check in checks),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    events.drop(columns=["onset_period"]).to_csv(
        output_dir / "onset_events.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    folds.to_csv(output_dir / "chronological_fold_counts.csv", index=False)
    samples.to_csv(
        output_dir / "eligible_forecast_samples.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the expanded SWBA and audit drought-onset "
            "labels and chronological sample counts."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Expanded monthly SWBA CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for audit report and CSV outputs.",
    )
    arguments = parser.parse_args()

    report = audit(arguments.input.resolve(), arguments.output_dir.resolve())
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["all_checks_passed"]:
        raise SystemExit("Data audit failed. Review audit_report.json.")


if __name__ == "__main__":
    main()
