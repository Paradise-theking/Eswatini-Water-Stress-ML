import unittest
from datetime import datetime
from types import ModuleType
from unittest.mock import patch

from backend.water_availability import (
    WaterRiskDataError,
    fetch_surface_water_area,
    summarize_surface_water_observations,
    summarize_weather_ensemble,
)


def build_weather_data(member_count=12):
    dates = [f"2026-01-{day:02d}" for day in range(1, 31)]
    daily = {"time": dates}
    for member in range(1, member_count + 1):
        daily[f"precipitation_sum_member{member}"] = [float(member)] * 30
        daily[f"et0_fao_evapotranspiration_member{member}"] = [
            member / 2
        ] * 30
    return {
        "daily": daily,
        "timezone": "Africa/Mbabane",
        "latitude": -26.0,
        "longitude": 31.5,
    }


class SummarizeWeatherEnsembleTests(unittest.TestCase):
    def test_summarizes_paired_member_totals_and_gross_rainfall(self):
        result = summarize_weather_ensemble(
            build_weather_data(),
            datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["ensemble_member_count"], 12)
        self.assertEqual(result["precipitation_total_mm"]["p10"], 63)
        self.assertEqual(result["precipitation_total_mm"]["median"], 195)
        self.assertEqual(result["net_climate_balance_mm"]["median"], 97.5)
        self.assertAlmostEqual(
            result["gross_rainfall_volume_ml"]["median"],
            195 * 896.9,
        )

    def test_rejects_wrong_number_of_forecast_dates(self):
        provider_data = build_weather_data()
        provider_data["daily"]["time"].pop()

        with self.assertRaisesRegex(WaterRiskDataError, "30 daily values"):
            summarize_weather_ensemble(provider_data)

    def test_requires_at_least_ten_complete_paired_members(self):
        provider_data = build_weather_data()
        provider_data["daily"]["precipitation_sum_member10"][0] = float("nan")
        provider_data["daily"]["precipitation_sum_member11"] = [1] * 29
        provider_data["daily"]["et0_fao_evapotranspiration_member12"] = [1] * 29

        with self.assertRaisesRegex(WaterRiskDataError, "fewer than 10"):
            summarize_weather_ensemble(provider_data)


class SummarizeSurfaceWaterObservationsTests(unittest.TestCase):
    def test_filters_low_coverage_and_deduplicates_dates(self):
        result = summarize_surface_water_observations([
            {
                "date": "2026-03-10",
                "water_area_m2": 2_000_000,
                "valid_coverage_fraction": 0.85,
            },
            {
                "date": "2026-03-10",
                "water_area_m2": 2_200_000,
                "valid_coverage_fraction": 0.9,
            },
            {
                "date": "2026-03-01",
                "water_area_m2": 1_000_000,
                "valid_coverage_fraction": 0.8,
            },
            {
                "date": "2026-03-12",
                "water_area_m2": 3_000_000,
                "valid_coverage_fraction": 0.79,
            },
        ])

        self.assertEqual(result["latest_observation"]["date"], "2026-03-10")
        self.assertEqual(result["latest_observation"]["water_area_km2"], 2.2)
        self.assertEqual(result["previous_observation"]["date"], "2026-03-01")
        self.assertAlmostEqual(
            result["area_change_percent_vs_previous_scene"],
            120,
        )

    def test_requires_a_qualifying_scene(self):
        with self.assertRaisesRegex(WaterRiskDataError, "No recent Sentinel-2"):
            summarize_surface_water_observations([{
                "date": "2026-03-10",
                "water_area_m2": 1_000_000,
                "valid_coverage_fraction": 0.79,
            }])

    def test_rejects_non_finite_area(self):
        with self.assertRaisesRegex(WaterRiskDataError, "out-of-range"):
            summarize_surface_water_observations([{
                "date": "2026-03-10",
                "water_area_m2": float("inf"),
                "valid_coverage_fraction": 0.9,
            }])


class FetchSurfaceWaterAreaTests(unittest.TestCase):
    def test_initializes_earth_engine_before_creating_geometry(self):
        initialized = False

        class GeometryReached(Exception):
            pass

        def initialize_earth_engine():
            nonlocal initialized
            initialized = True

        def create_geometry(_geometry):
            self.assertTrue(initialized)
            raise GeometryReached

        ee_module = ModuleType("ee")
        ee_module.Geometry = create_geometry
        ingestion_module = ModuleType("backend.data_ingestion")
        ingestion_module.initialize_earth_engine = initialize_earth_engine

        with patch.dict(
            "sys.modules",
            {
                "ee": ee_module,
                "backend.data_ingestion": ingestion_module,
            },
        ):
            with self.assertRaises(GeometryReached):
                fetch_surface_water_area()


if __name__ == "__main__":
    unittest.main()
