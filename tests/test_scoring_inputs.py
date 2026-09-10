import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from score import (  # noqa: E402
    compute_and_attach_metriq_scores,
    compute_device_composite_scores,
)


BENCHMARK = "Example"
KEY = (BENCHMARK, "score", "null")
CONFIG = {
    "default": {
        "baseline": {"provider": "ibm", "device": "base"},
        "composite": {
            "components": [{
                "label": BENCHMARK,
                "weight": 1,
                "components": [{
                    "benchmark": BENCHMARK,
                    "metric": "score",
                    "label": "measurement",
                    "weight": 1,
                }],
            }],
        },
    },
}


def row(value, *, device="device", timestamp="2026-01-01T00:00:00Z", direction="higher"):
    return {
        "provider": "ibm",
        "device": device,
        "timestamp": timestamp,
        "job_type": BENCHMARK,
        "results": {"score": value},
        "directions": {"score": direction},
    }


class ScoringInputTests(unittest.TestCase):
    def test_raw_baseline_and_direction_follow_latest_raw_not_normalized_timestamp(self):
        for latest_baseline in (None, 0.0):
            with self.subTest(latest_baseline=latest_baseline):
                older = row(0.8)
                newer = row(0.0, timestamp="2026-02-01T00:00:00Z", direction="lower")
                rows = [older, newer]
                series = {id(older): "v0.6", id(newer): "v0.7"}
                baselines = {"v0.6": {KEY: 0.4}, "v0.7": {}}
                if latest_baseline is not None:
                    baselines["v0.7"][KEY] = latest_baseline
                compute_and_attach_metriq_scores(rows, series, baselines, CONFIG)
                record = compute_device_composite_scores(rows, series, baselines, CONFIG)[0]
                component = record["components"]["measurement"]

                self.assertEqual(component["raw"], 0.0)
                self.assertEqual(component["raw_timestamp"], newer["timestamp"])
                self.assertEqual(component["baseline"], latest_baseline)
                self.assertEqual(component["direction"], "lower")
                self.assertFalse(component["baseline_is_self"])
                self.assertEqual(component["normalized"], 200.0)
                self.assertEqual(component["normalized_timestamp"], older["timestamp"])
                self.assertEqual(newer["normalization_baselines"], {"score": latest_baseline})

    def test_baseline_composite_anchors_to_selected_raw_without_changing_row_normalization(self):
        baseline_row = row(0.2, device="base")
        series = {id(baseline_row): "v0.7"}
        baselines = {"v0.7": {KEY: 0.4}}
        compute_and_attach_metriq_scores([baseline_row], series, baselines, CONFIG)
        record = compute_device_composite_scores([baseline_row], series, baselines, CONFIG)[0]
        component = record["components"]["measurement"]

        self.assertEqual(baseline_row["normalization_baselines"], {"score": 0.4})
        self.assertEqual(baseline_row["normalized_scores"], {"score": 50.0})
        self.assertEqual(component["baseline"], 0.2)
        self.assertEqual(component["normalized"], 100.0)
        self.assertTrue(component["baseline_is_self"])
        self.assertEqual(record["metriq_score"], 100.0)

    def test_raw_self_flag_wins_when_normalized_row_used_another_baseline(self):
        older = row(0.8, device="base")
        newer = row(0.0, device="base", timestamp="2026-02-01T00:00:00Z", direction="lower")
        config = CONFIG | {
            "series": {
                "v0.7": {"baseline": {"provider": "ibm", "device": "another_base"}},
            },
        }
        series = {id(older): "v0.6", id(newer): "v0.7"}
        baselines = {"v0.6": {KEY: 0.4}, "v0.7": {KEY: 0.2}}
        record = compute_device_composite_scores([older, newer], series, baselines, config)[0]
        component = record["components"]["measurement"]

        self.assertEqual(component["normalized"], 100.0)
        self.assertEqual(component["normalized_timestamp"], older["timestamp"])
        self.assertEqual(component["baseline"], 0.2)
        self.assertEqual(component["raw_timestamp"], newer["timestamp"])
        self.assertFalse(component["baseline_is_self"])

    def test_zero_self_baseline_remains_a_raw_measurement(self):
        baseline_row = row(0.0, device="base")
        series = {id(baseline_row): "v0.7"}
        baselines = {"v0.7": {KEY: 0.0}}
        compute_and_attach_metriq_scores([baseline_row], series, baselines, CONFIG)
        record = compute_device_composite_scores([baseline_row], series, baselines, CONFIG)[0]
        component = record["components"]["measurement"]

        self.assertEqual(baseline_row["normalization_baselines"], {"score": 0.0})
        self.assertNotIn("normalized_scores", baseline_row)
        self.assertIsNone(component["normalized"])
        self.assertTrue(component["raw_available"])
        self.assertEqual(component["baseline"], 0.0)
        self.assertTrue(component["baseline_is_self"])

    def test_lower_is_better_inputs_reproduce_normalization(self):
        device_row = row(0.002, direction="lower")
        series = {id(device_row): "v0.7"}
        baselines = {"v0.7": {KEY: 0.005}}
        compute_and_attach_metriq_scores([device_row], series, baselines, CONFIG)
        record = compute_device_composite_scores([device_row], series, baselines, CONFIG)[0]
        component = record["components"]["measurement"]

        self.assertEqual(device_row["normalization_baselines"], {"score": 0.005})
        self.assertEqual(device_row["normalized_scores"], {"score": 250.0})
        self.assertEqual(component["baseline"], 0.005)
        self.assertEqual(component["direction"], "lower")
        self.assertFalse(component["baseline_is_self"])
        self.assertEqual(record["metriq_score"], 250.0)

    def test_missing_component_has_no_reference_or_self_anchor(self):
        device_row = row(None, device="base")
        series = {id(device_row): "v0.7"}
        record = compute_device_composite_scores([device_row], series, {}, CONFIG)[0]
        component = record["components"]["measurement"]

        self.assertIsNone(component["raw"])
        self.assertIsNone(component["baseline"])
        self.assertFalse(component["baseline_is_self"])

    def test_derived_score_has_no_raw_normalization_baseline(self):
        config = {
            "default": {"composite": {"components": [{
                "benchmark": BENCHMARK,
                "metric": "derived_score",
                "weight": 1,
                "derived_from": [{"metric": "score", "weight": 1}],
            }]}},
        }
        device_row = row(0.8)
        series = {id(device_row): "v0.7"}
        baselines = {"v0.7": {KEY: 0.4}}
        compute_and_attach_metriq_scores([device_row], series, baselines, config)

        self.assertEqual(device_row["normalized_scores"], {"derived_score": 200.0})
        self.assertEqual(device_row["normalization_baselines"], {"derived_score": None})

    def test_normalized_only_anchor_comes_from_selected_normalized_record(self):
        older = row(None, device="base")
        older["normalized_scores"] = {"score": 250.0}
        newer = row(None, device="base", timestamp="2026-02-01T00:00:00Z")
        newer["normalized_scores"] = {"score": 200.0}
        config = CONFIG | {
            "series": {
                "v0.7": {"baseline": {"provider": "ibm", "device": "another_base"}},
            },
        }
        series = {id(older): "v0.6", id(newer): "v0.7"}
        for rows, expected, is_self in (([older], 100.0, True), ([newer, older], 200.0, False)):
            with self.subTest(expected=expected):
                record = compute_device_composite_scores(rows, series, {}, config)[0]
                component = record["components"]["measurement"]
                self.assertIsNone(component["baseline"])
                self.assertEqual(component["normalized"], expected)
                self.assertEqual(component["baseline_is_self"], is_self)


if __name__ == "__main__":
    unittest.main()
