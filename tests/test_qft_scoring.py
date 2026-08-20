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
)


BENCH = "Quantum Fourier Transform"
WIDTHS = [4, 8, 12, 20]
WEIGHTS = [1 / 11, 2 / 11, 3 / 11, 5 / 11]


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
    }


def _scoring_cfg():
    return {
        "default": {
            "baseline": {"provider": "ibm", "device": "ibm_torino"},
            "composite": {"components": [_qft_group()]},
        }
    }


def _row(device, width, value, *, legacy):
    params = (
        {"min_qubits": width, "max_qubits": width, "skip_qubits": 1}
        if legacy
        else {"num_qubits": width}
    )
    return {
        "provider": "ibm",
        "device": device,
        "timestamp": f"2026-01-01T00:00:{width:02d}",
        "job_type": BENCH,
        "params": params,
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

    def test_published_qft_panels_follow_the_score_definition(self):
        with (ROOT / "scripts" / "scoring.json").open(encoding="utf-8") as f:
            scoring_cfg = json.load(f)

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

        legacy_qft_group = next(
            group
            for group in scoring_cfg["series"]["v0.4"]["composite"]["components"]
            if group.get("label") == BENCH
        )
        self.assertEqual(len(legacy_qft_group["components"]), 1)
        self.assertNotIn("selector", legacy_qft_group["components"][0])


if __name__ == "__main__":
    unittest.main()
