import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from etl import upsert_platform, write_platform_outputs  # noqa: E402
from score import (  # noqa: E402
    _flatten_components,
    compute_device_composite_scores,
    load_scoring_config,
    validate_scoring_config,
)


def scoring_config(*, suite="future_score_2_0", component="future-component"):
    composite = {
        "components": [
            {
                "label": "Future component",
                "weight": "1/1",
                "components": [
                    {
                        "benchmark": "Future benchmark",
                        "metric": "score",
                        "label": "Future-10",
                        "selector": {"num_qubits": 10},
                        "weight": "1/2",
                    },
                    {
                        "benchmark": "Future benchmark",
                        "metric": "score",
                        "label": "Future-20",
                        "selector": {"num_qubits": 20},
                        "weight": "1/2",
                    },
                ],
            }
        ]
    }
    if suite is not None:
        composite["dispatch"] = {"suite": suite}
    if component is not None:
        composite["components"][0]["dispatch"] = {"component": component}
    return {"default": {"composite": composite}}


def composite_record(config, series="v0.7"):
    row = {
        "provider": "ibm",
        "device": "ibm_future",
        "timestamp": "2026-08-13T12:00:00Z",
        "benchmark": "Unrelated benchmark",
        "results": {},
    }
    return compute_device_composite_scores(
        [row],
        {id(row): series},
        {},
        config,
    )[0]


class ScoringDispatchTests(unittest.TestCase):
    def test_dispatch_metadata_is_published_for_every_leaf_including_missing_results(self):
        config = scoring_config()
        validate_scoring_config(config)

        record = composite_record(config)

        self.assertEqual(record["series"], "v0.7")
        self.assertEqual(set(record["components"]), {"Future-10", "Future-20"})
        for component in record["components"].values():
            self.assertFalse(component["raw_available"])
            self.assertFalse(component["normalized_available"])
            self.assertEqual(
                component["dispatch"],
                {"suite": "future_score_2_0", "component": "future-component"},
            )

    def test_config_without_dispatch_metadata_keeps_the_previous_output_shape(self):
        config = scoring_config(suite=None, component=None)
        validate_scoring_config(config)

        record = composite_record(config)

        for component in record["components"].values():
            self.assertNotIn("dispatch", component)

    def test_series_specific_composite_can_select_a_different_suite_version(self):
        config = scoring_config(suite="future_score_2_0", component="future-component")
        historical = scoring_config(
            suite="historical_score_1_0",
            component="historical-component",
        )["default"]
        config["series"] = {"v0.4": historical}
        validate_scoring_config(config)

        historical_record = composite_record(config, series="v0.4")
        current_record = composite_record(config, series="v0.7")

        self.assertEqual(
            historical_record["components"]["Future-10"]["dispatch"],
            {"suite": "historical_score_1_0", "component": "historical-component"},
        )
        self.assertEqual(
            current_record["components"]["Future-10"]["dispatch"],
            {"suite": "future_score_2_0", "component": "future-component"},
        )

    def test_validation_rejects_incomplete_or_malformed_dispatch_metadata(self):
        cases = []

        suite_only = scoring_config(component=None)
        cases.append(suite_only)

        component_only = scoring_config(suite=None)
        cases.append(component_only)

        empty_suite = scoring_config(suite="   ")
        cases.append(empty_suite)

        non_string_component = scoring_config()
        non_string_component["default"]["composite"]["components"][0]["dispatch"] = {
            "component": 42
        }
        cases.append(non_string_component)

        non_object = scoring_config()
        non_object["default"]["composite"]["dispatch"] = "future_score_2_0"
        cases.append(non_object)

        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    validate_scoring_config(config)

    def test_checked_in_series_resolve_complete_dispatch_pairs(self):
        config = load_scoring_config(str(REPO_ROOT))
        validate_scoring_config(config)

        composites = [
            config["series"]["v0.4"]["composite"],
            config["default"]["composite"],
        ]
        for composite in composites:
            suite = composite["dispatch"]["suite"]
            flat = _flatten_components(composite["components"])
            self.assertTrue(flat)
            for leaf in flat:
                self.assertEqual(suite, "metriq_score_1_0")
                self.assertIsInstance(leaf["_dispatch"]["component"], str)
                self.assertTrue(leaf["_dispatch"]["component"])

    def test_platform_output_preserves_dispatch_metadata(self):
        config = scoring_config()
        record = composite_record(config)
        source_row = {
            "timestamp": "2026-08-13T12:00:00Z",
            "platform": {"provider": "ibm", "device": "ibm_future"},
            "results": {},
        }
        registry = {}
        upsert_platform(registry, source_row, "result.json", "v0.7")

        with tempfile.TemporaryDirectory() as tmp:
            write_platform_outputs(
                registry,
                tmp,
                "2026-08-13T12:30:00Z",
                composite_records=[record],
            )
            payload = json.loads(
                (Path(tmp) / "platforms" / "ibm" / "ibm_future.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            payload["metriq_score"]["components"]["Future-10"]["dispatch"],
            {"suite": "future_score_2_0", "component": "future-component"},
        )


if __name__ == "__main__":
    unittest.main()
