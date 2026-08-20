import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from score import (  # noqa: E402
    _component_param_matches,
    _parse_weight,
    _row_param_matches,
    _selector_fingerprint,
    compute_and_attach_metriq_scores,
    compute_baseline_averages_by_series,
    compute_device_composite_scores,
    validate_scoring_config,
)


BENCH = "Quantum Fourier Transform"
WIDTHS = [4, 8, 12, 20]
WEIGHTS = [1 / 11, 2 / 11, 3 / 11, 5 / 11]


def _legacy_baseline_component(width, weight):
    return {
        "selector": {"num_qubits": width},
        "selector_alternatives": [
            {"min_qubits": width, "max_qubits": width}
        ],
        "weight": weight,
    }


def _aggregate_fallbacks():
    return [
        {
            "benchmark": BENCH,
            "metric": "score",
            "selector": {"min_qubits": 4, "max_qubits": 12, "skip_qubits": 4},
            "label": "QFT (legacy 4–12 sweep):score",
            "covers": ["QFT-4:score", "QFT-8:score", "QFT-12:score"],
            "required_num_qubits": 12,
            "baseline_components": [
                _legacy_baseline_component(width, "1/3")
                for width in (4, 8, 12)
            ],
        },
        {
            "benchmark": BENCH,
            "metric": "score",
            "selector": {"min_qubits": 4, "max_qubits": 20, "skip_qubits": 4},
            "label": "QFT (legacy 4–20 sweep):score",
            "covers": [
                "QFT-4:score",
                "QFT-8:score",
                "QFT-12:score",
                "QFT-20:score",
            ],
            "required_num_qubits": 20,
            "baseline_components": [
                _legacy_baseline_component(width, "1/5")
                for width in (4, 8, 12, 16, 20)
            ],
        },
    ]


def _qft_group(weight="1"):
    return {
        "label": BENCH,
        "weight": weight,
        "components": [
            {
                "benchmark": BENCH,
                "metric": "score",
                "selector": {"num_qubits": width},
                "selector_alternatives": [
                    {"min_qubits": width, "max_qubits": width}
                ],
                "label": f"QFT-{width}:score",
                "weight": f"{width}/44",
            }
            for width in WIDTHS
        ],
        "aggregate_fallbacks": _aggregate_fallbacks(),
    }


def _scoring_cfg():
    return {
        "default": {
            "baseline": {"provider": "ibm", "device": "ibm_torino"},
            "composite": {"components": [_qft_group()]},
        }
    }


def _row(device, width, value, *, legacy, provider="ibm"):
    params = (
        {"min_qubits": width, "max_qubits": width, "skip_qubits": 1}
        if legacy
        else {"num_qubits": width}
    )
    return {
        "provider": provider,
        "device": device,
        "timestamp": f"2026-01-01T00:00:{width:02d}",
        "job_type": BENCH,
        "params": params,
        "results": {"score": value},
        "directions": {"score": "higher"},
    }


def _aggregate_row(
    device,
    max_qubits,
    value,
    *,
    provider="quantinuum",
    timestamp="2025-12-08T10:03:08.711173",
):
    return {
        "provider": provider,
        "device": device,
        "timestamp": timestamp,
        "job_type": BENCH,
        "params": {
            "min_qubits": 4,
            "max_qubits": max_qubits,
            "skip_qubits": 4,
            "max_circuits": 3,
        },
        "results": {"score": value},
        "directions": {"score": "higher"},
    }


class QFTParameterCompatibilityTests(unittest.TestCase):
    def test_num_qubits_selector_matches_current_schema(self):
        row = _row("device", 8, 0.5, legacy=False)
        self.assertTrue(_component_param_matches(_qft_group()["components"][1], row))
        self.assertFalse(_component_param_matches(_qft_group()["components"][2], row))

    def test_num_qubits_selector_matches_legacy_fixed_width(self):
        row = _row("device", 8, 0.5, legacy=True)
        component = _qft_group()["components"][1]
        self.assertFalse(_row_param_matches(component["selector"], row))
        self.assertTrue(_component_param_matches(component, row))
        self.assertFalse(_component_param_matches(_qft_group()["components"][2], row))

    def test_legacy_multi_width_result_is_not_misrepresented_as_one_width(self):
        row = _row("device", 8, 0.5, legacy=True)
        row["params"].update({"min_qubits": 4, "max_qubits": 12, "skip_qubits": 2})
        for component in _qft_group()["components"][:3]:
            self.assertFalse(_component_param_matches(component, row))

    def test_alternative_selectors_are_benchmark_agnostic(self):
        component = {
            "selector": {"current_size": 8},
            "selector_alternatives": [{"legacy_min": 8, "legacy_max": 8}],
        }
        row = {"params": {"legacy_min": 8, "legacy_max": 8}}
        self.assertTrue(_component_param_matches(component, row))


class QFTCompositeTests(unittest.TestCase):
    def test_mixed_legacy_and_current_rows_use_all_canonical_widths(self):
        baseline_values = [0.808, 0.215, 0.002, 0.0]
        device_values = [0.862, 0.420, 0.038, 0.0]
        rows = [
            *[
                _row("ibm_torino", width, value, legacy=True)
                for width, value in zip(WIDTHS, baseline_values)
            ],
            *[
                _row("ibm_boston", width, value, legacy=False)
                for width, value in zip(WIDTHS, device_values)
            ],
        ]
        row_series = {id(row): "v0.7" for row in rows}
        scoring_cfg = _scoring_cfg()
        baseline_avg, _ = compute_baseline_averages_by_series(
            rows, row_series, scoring_cfg
        )
        baseline_keys = baseline_avg["v0.7"]
        for width, value in zip(WIDTHS, baseline_values):
            canonical_key = (
                BENCH,
                "score",
                _selector_fingerprint({"num_qubits": width}),
            )
            self.assertEqual(baseline_keys[canonical_key], value)
            self.assertNotIn(
                (
                    BENCH,
                    "score",
                    _selector_fingerprint(
                        {"min_qubits": width, "max_qubits": width}
                    ),
                ),
                baseline_keys,
            )
        compute_and_attach_metriq_scores(
            rows, row_series, baseline_avg, scoring_cfg
        )

        records = {
            record["device"]: record
            for record in compute_device_composite_scores(
                rows, row_series, baseline_avg, scoring_cfg
            )
        }
        baseline_raw = sum(w * value for w, value in zip(WEIGHTS, baseline_values))
        device_raw = sum(w * value for w, value in zip(WEIGHTS, device_values))

        self.assertAlmostEqual(records["ibm_torino"]["metriq_score"], 100.0)
        self.assertAlmostEqual(
            records["ibm_boston"]["metriq_score"],
            100.0 * device_raw / baseline_raw,
        )
        self.assertEqual(len(records["ibm_boston"]["components"]), 4)
        self.assertEqual(
            [
                records["ibm_boston"]["components"][f"QFT-{width}:score"][
                    "required_num_qubits"
                ]
                for width in WIDTHS
            ],
            WIDTHS,
        )
        self.assertTrue(
            records["ibm_boston"]["components"]["QFT-20:score"]["raw_available"]
        )

    def test_quantinuum_legacy_sweep_is_scored_without_inventing_width_values(self):
        baseline_values = {4: 0.808, 8: 0.215, 12: 0.002, 20: 0.0}
        baseline_rows = [
            _row("ibm_torino", width, value, legacy=True)
            for width, value in baseline_values.items()
        ]
        quantinuum_row = _aggregate_row("H2-2", 12, 0.967)
        rows = [*baseline_rows, quantinuum_row]
        row_series = {
            **{id(row): "v0.6" for row in baseline_rows},
            id(quantinuum_row): "v0.5",
        }
        scoring_cfg = _scoring_cfg()
        baseline_avg, _ = compute_baseline_averages_by_series(
            rows, row_series, scoring_cfg
        )
        compute_and_attach_metriq_scores(
            rows, row_series, baseline_avg, scoring_cfg
        )

        records = {
            record["device"]: record
            for record in compute_device_composite_scores(
                rows, row_series, baseline_avg, scoring_cfg
            )
        }
        quantinuum = records["H2-2"]
        fallback_label = "QFT (legacy 4–12 sweep):score"
        component = quantinuum["components"][fallback_label]
        baseline_sweep = (0.808 + 0.215 + 0.002) / 3
        normalized = 100.0 * 0.967 / baseline_sweep
        coverage = 6 / 11

        self.assertAlmostEqual(quantinuum_row["normalized_scores"]["score"], normalized)
        self.assertAlmostEqual(quantinuum_row["metriq_score"], normalized)
        self.assertAlmostEqual(component["raw"], 0.967)
        self.assertAlmostEqual(component["normalized"], normalized)
        self.assertAlmostEqual(component["coverage"], coverage)
        self.assertAlmostEqual(component["weight"], coverage)
        self.assertEqual(component["required_num_qubits"], 12)
        self.assertTrue(component["aggregate_fallback"])
        self.assertEqual(
            component["covered_components"],
            ["QFT-4:score", "QFT-8:score", "QFT-12:score"],
        )
        self.assertNotIn("QFT-4:score", quantinuum["components"])
        self.assertNotIn("QFT-8:score", quantinuum["components"])
        self.assertNotIn("QFT-12:score", quantinuum["components"])
        self.assertFalse(
            quantinuum["components"]["QFT-20:score"]["raw_available"]
        )
        self.assertAlmostEqual(quantinuum["metriq_score"], normalized * coverage)

    def test_canonical_width_data_takes_precedence_over_legacy_sweep(self):
        baseline_rows = [
            _row("ibm_torino", width, value, legacy=True)
            for width, value in zip(WIDTHS, [0.808, 0.215, 0.002, 0.0])
        ]
        quantinuum_sweep = _aggregate_row("H2-2", 12, 0.967)
        quantinuum_width = _row(
            "H2-2", 4, 0.9, legacy=False, provider="quantinuum"
        )
        rows = [*baseline_rows, quantinuum_sweep, quantinuum_width]
        row_series = {id(row): "v0.6" for row in rows}
        scoring_cfg = _scoring_cfg()
        baseline_avg, _ = compute_baseline_averages_by_series(
            rows, row_series, scoring_cfg
        )

        quantinuum = next(
            record
            for record in compute_device_composite_scores(
                rows, row_series, baseline_avg, scoring_cfg
            )
            if record["device"] == "H2-2"
        )

        self.assertNotIn(
            "QFT (legacy 4–12 sweep):score", quantinuum["components"]
        )
        self.assertEqual(quantinuum["components"]["QFT-4:score"]["raw"], 0.9)

    def test_widest_legacy_sweep_wins_and_uses_its_full_baseline(self):
        baseline_values = {4: 0.8, 8: 0.4, 12: 0.2, 16: 0.1, 20: 0.05}
        baseline_rows = [
            _row("ibm_torino", width, value, legacy=True)
            for width, value in baseline_values.items()
        ]
        full_sweep = _aggregate_row(
            "legacy-device",
            20,
            0.5,
            provider="origin",
            timestamp="2026-01-01T00:00:01",
        )
        newer_partial_sweep = _aggregate_row(
            "legacy-device",
            12,
            0.9,
            provider="origin",
            timestamp="2026-01-02T00:00:01",
        )
        rows = [*baseline_rows, full_sweep, newer_partial_sweep]
        row_series = {id(row): "v0.6" for row in rows}
        scoring_cfg = _scoring_cfg()
        baseline_avg, _ = compute_baseline_averages_by_series(
            rows, row_series, scoring_cfg
        )

        record = next(
            record
            for record in compute_device_composite_scores(
                rows, row_series, baseline_avg, scoring_cfg
            )
            if record["device"] == "legacy-device"
        )
        component = record["components"]["QFT (legacy 4–20 sweep):score"]
        expected_baseline = sum(baseline_values.values()) / 5

        self.assertNotIn("QFT (legacy 4–12 sweep):score", record["components"])
        self.assertAlmostEqual(component["normalized"], 100.0 * 0.5 / expected_baseline)
        self.assertAlmostEqual(component["coverage"], 1.0)
        self.assertEqual(component["required_num_qubits"], 20)

    def test_published_qft_panels_follow_the_score_definition(self):
        with (ROOT / "scripts" / "scoring.json").open(encoding="utf-8") as f:
            scoring_cfg = json.load(f)

        validate_scoring_config(scoring_cfg)

        for block_name in ("default", "v1.0"):
            block = (
                scoring_cfg["default"]
                if block_name == "default"
                else scoring_cfg["series"][block_name]
            )
            qft_group = next(
                group
                for group in block["composite"]["components"]
                if group.get("label") == BENCH
            )
            self.assertEqual(
                [child["selector"]["num_qubits"] for child in qft_group["components"]],
                WIDTHS,
            )
            self.assertEqual(
                [
                    child["selector_alternatives"]
                    for child in qft_group["components"]
                ],
                [
                    [{"min_qubits": width, "max_qubits": width}]
                    for width in WIDTHS
                ],
            )
            self.assertEqual(
                [_parse_weight(child["weight"]) for child in qft_group["components"]],
                WEIGHTS,
            )
            self.assertEqual(
                [
                    fallback["selector"]
                    for fallback in qft_group["aggregate_fallbacks"]
                ],
                [
                    {"min_qubits": 4, "max_qubits": 12, "skip_qubits": 4},
                    {"min_qubits": 4, "max_qubits": 20, "skip_qubits": 4},
                ],
            )
            self.assertEqual(
                [fallback["covers"] for fallback in qft_group["aggregate_fallbacks"]],
                [
                    ["QFT-4:score", "QFT-8:score", "QFT-12:score"],
                    [
                        "QFT-4:score",
                        "QFT-8:score",
                        "QFT-12:score",
                        "QFT-20:score",
                    ],
                ],
            )

        legacy_qft_group = next(
            group
            for group in scoring_cfg["series"]["v0.4"]["composite"]["components"]
            if group.get("label") == BENCH
        )
        self.assertEqual(len(legacy_qft_group["components"]), 1)
        self.assertNotIn("selector", legacy_qft_group["components"][0])


class AggregateFallbackValidationTests(unittest.TestCase):
    def test_covered_components_must_reference_group_children(self):
        scoring_cfg = _scoring_cfg()
        scoring_cfg["default"]["composite"]["components"][0][
            "aggregate_fallbacks"
        ][0]["covers"] = ["not-a-child"]

        with self.assertRaisesRegex(ValueError, "Invalid covered components"):
            validate_scoring_config(scoring_cfg)

    def test_baseline_component_weights_must_sum_to_one(self):
        scoring_cfg = _scoring_cfg()
        scoring_cfg["default"]["composite"]["components"][0][
            "aggregate_fallbacks"
        ][0]["baseline_components"][0]["weight"] = "1/2"

        with self.assertRaisesRegex(
            ValueError, "Baseline component weights must sum to 1.0"
        ):
            validate_scoring_config(scoring_cfg)

    def test_overlapping_fallbacks_for_one_metric_are_rejected_at_runtime(self):
        scoring_cfg = _scoring_cfg()
        fallbacks = scoring_cfg["default"]["composite"]["components"][0][
            "aggregate_fallbacks"
        ]
        overlapping = json.loads(json.dumps(fallbacks[0]))
        overlapping["selector"] = {"min_qubits": 4}
        overlapping["label"] = "overlapping aggregate"
        fallbacks.append(overlapping)

        baseline_rows = [
            _row("ibm_torino", width, value, legacy=True)
            for width, value in zip(WIDTHS, [0.808, 0.215, 0.002, 0.0])
        ]
        aggregate_row = _aggregate_row("H2-2", 12, 0.967)
        rows = [*baseline_rows, aggregate_row]
        row_series = {id(row): "v0.6" for row in rows}
        baseline_avg, _ = compute_baseline_averages_by_series(
            rows, row_series, scoring_cfg
        )

        with self.assertRaisesRegex(ValueError, "Ambiguous aggregate fallbacks"):
            compute_and_attach_metriq_scores(
                rows, row_series, baseline_avg, scoring_cfg
            )


if __name__ == "__main__":
    unittest.main()
