import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from etl import InvalidRecordOutcome, flatten_row, validate_record_outcome  # noqa: E402
from score import _selector_fingerprint, compute_device_composite_scores  # noqa: E402


CEPHEUS_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"


def outcome_record(**overrides):
    record = {
        "app_version": "0.7.2",
        "timestamp": "2026-08-07T12:00:00",
        "job_type": "Linear Ramp QAOA",
        "params": {"benchmark_name": "Linear Ramp QAOA", "num_qubits": 100},
        "platform": {"provider": "braket", "device": CEPHEUS_ARN},
        "outcome": "unsupported",
        "outcome_detail": {
            "reason": "Compiler rejects 100-qubit LR-QAOA circuits",
            "error_message": "compilation failed: too many qubits in routing",
            "source": "dispatch",
        },
        "results": None,
    }
    record.update(overrides)
    return record


class RecordOutcomeTests(unittest.TestCase):
    def test_outcome_and_detail_pass_through(self):
        out = flatten_row(outcome_record())
        self.assertEqual(out["outcome"], "unsupported")
        self.assertEqual(
            out["outcome_detail"]["reason"], "Compiler rejects 100-qubit LR-QAOA circuits"
        )
        self.assertEqual(out["outcome_detail"]["source"], "dispatch")
        self.assertEqual(out["provider"], "aws")
        self.assertEqual(out["device"], "rigetti_cepheus-1-108q")
        self.assertEqual(out["params"]["num_qubits"], 100)

    def test_completed_outcome_is_implied_not_stamped(self):
        out = flatten_row(outcome_record(outcome="completed"))
        self.assertNotIn("outcome", out)
        self.assertNotIn("outcome_detail", out)

    def test_records_without_outcome_are_unchanged(self):
        out = flatten_row(
            {
                "timestamp": "2026-08-07T12:00:00",
                "job_type": "WIT",
                "platform": {"provider": "ibm", "device": "ibm_fez"},
                "results": {"score": {"value": 88.4, "uncertainty": 0.1}},
            }
        )
        self.assertNotIn("outcome", out)
        self.assertEqual(out["results"]["score"], 88.4)

    def test_unknown_outcome_values_are_dropped_by_flatten(self):
        # validate_record_outcome rejects these before flatten_row runs in the
        # ETL; flatten_row itself stays defensive.
        out = flatten_row(outcome_record(outcome="exploded"))
        self.assertNotIn("outcome", out)
        self.assertNotIn("outcome_detail", out)

    def test_outcome_detail_is_normalized(self):
        out = flatten_row(
            outcome_record(
                outcome="error",
                outcome_detail={"reason": "  queue died  ", "unknown_key": "x", "source": ""},
            )
        )
        self.assertEqual(out["outcome"], "error")
        self.assertEqual(out["outcome_detail"], {"reason": "queue died"})

    def test_outcome_without_detail(self):
        out = flatten_row(outcome_record(outcome_detail=None))
        self.assertEqual(out["outcome"], "unsupported")
        self.assertNotIn("outcome_detail", out)

    def test_non_completed_outcomes_drop_results_payloads(self):
        out = flatten_row(
            outcome_record(outcome="error", results={"score": {"value": 1.23, "uncertainty": 0.01}})
        )
        self.assertEqual(out["outcome"], "error")
        self.assertIn("results", out)
        self.assertIsNone(out["results"])


# Records exactly as produced by metriq-gym's upload path
# (unitaryfoundation/metriq-gym#805): `mgym job upload --outcome unsupported
# --reason ...` on a dispatch-failed job, and a plain failed-job upload whose
# job has no captured error (outcome_detail is emitted as null).
MGYM_UNSUPPORTED_RECORD = {
    "app_version": "0.7.2.dev17",
    "timestamp": "2026-08-19T12:46:48",
    "job_type": "Linear Ramp QAOA",
    "results": None,
    "outcome": "unsupported",
    "outcome_detail": {
        "reason": "Compiler rejects 100-qubit LR-QAOA circuits",
        "error_message": "RuntimeError: Error occurred during compilation: too many qubits in routing",
        "source": "dispatch",
    },
    "platform": {"device": "rigetti_cepheus-1-108q", "provider": "aws"},
    "params": {"benchmark_name": "Linear Ramp QAOA", "graph_type": "1D", "num_qubits": 100},
}
MGYM_ERROR_RECORD_NO_DETAIL = {
    "app_version": "0.7.2.dev17",
    "timestamp": "2026-08-19T13:00:00",
    "job_type": "WIT",
    "results": None,
    "outcome": "error",
    "outcome_detail": None,
    "platform": {"device": "ibm_fez", "provider": "ibm"},
    "params": {"benchmark_name": "WIT", "num_qubits": 4},
}


class MetriqGymUploadCompatibilityTests(unittest.TestCase):
    def test_mgym_unsupported_record(self):
        self.assertEqual(validate_record_outcome(MGYM_UNSUPPORTED_RECORD), [])
        out = flatten_row(MGYM_UNSUPPORTED_RECORD)
        self.assertEqual(out["outcome"], "unsupported")
        self.assertIsNone(out["results"])
        self.assertEqual(out["outcome_detail"], MGYM_UNSUPPORTED_RECORD["outcome_detail"])
        self.assertEqual((out["provider"], out["device"]), ("aws", "rigetti_cepheus-1-108q"))

    def test_mgym_error_record_with_null_detail(self):
        self.assertEqual(validate_record_outcome(MGYM_ERROR_RECORD_NO_DETAIL), [])
        out = flatten_row(MGYM_ERROR_RECORD_NO_DETAIL)
        self.assertEqual(out["outcome"], "error")
        self.assertIsNone(out["results"])
        self.assertNotIn("outcome_detail", out)

    def test_mgym_records_are_stamped_on_components(self):
        rows = [flatten_row(MGYM_UNSUPPORTED_RECORD)]
        comp = _composite(rows)["rigetti_cepheus-1-108q"]["components"]["LR-QAOA 100q"]
        self.assertEqual(comp["reported_outcome"], "unsupported")
        self.assertEqual(comp["reported_outcome_reason"], "Compiler rejects 100-qubit LR-QAOA circuits")
        self.assertEqual(comp["reported_outcome_timestamp"], "2026-08-19T12:46:48")


class RecordOutcomeValidationTests(unittest.TestCase):
    def test_valid_outcome_record(self):
        self.assertEqual(validate_record_outcome(outcome_record()), [])

    def test_valid_completed_record(self):
        self.assertEqual(
            validate_record_outcome({"job_type": "WIT", "results": {"score": 1}}), []
        )

    def test_outcome_record_with_results_warns(self):
        msgs = validate_record_outcome(outcome_record(results={"score": 1}))
        self.assertEqual(len(msgs), 1)
        self.assertIn("results will be dropped", msgs[0])

    def test_outcome_record_without_params_warns(self):
        msgs = validate_record_outcome(outcome_record(params={}))
        self.assertTrue(any("no params" in m for m in msgs))

    def test_completed_record_without_results_warns(self):
        msgs = validate_record_outcome({"job_type": "WIT", "results": None})
        self.assertEqual(len(msgs), 1)
        self.assertIn("has no results", msgs[0])

    def test_invalid_outcome_is_an_error(self):
        for bad in ("exploded", "Unsupported", " error ", "", 1, True, {"x": 1}):
            with self.subTest(outcome=bad):
                with self.assertRaises(InvalidRecordOutcome):
                    validate_record_outcome(outcome_record(outcome=bad))

    def test_all_allowed_outcomes_accepted(self):
        for ok in ("completed", "error", "unsupported", "not_applicable"):
            with self.subTest(outcome=ok):
                rec = outcome_record(outcome=ok)
                if ok == "completed":
                    rec["results"] = {"score": 1}
                self.assertEqual(validate_record_outcome(rec), [])

    def test_absent_outcome_is_completed(self):
        self.assertEqual(validate_record_outcome({"results": {"score": 1}}), [])


LRQAOA = "Linear Ramp QAOA"
WIT = "WIT"
SERIES = "v0.7"


def _scoring_cfg():
    return {
        "default": {
            "baseline": {"provider": "ibm", "device": "base_device"},
            "composite": {
                "components": [
                    {
                        "benchmark": LRQAOA,
                        "metric": "score",
                        "selector": {"num_qubits": 100},
                        "label": "LR-QAOA 100q",
                        "weight": "1/2",
                    },
                    {
                        "benchmark": WIT,
                        "metric": "score",
                        "label": "WIT",
                        "weight": "1/2",
                    },
                ]
            },
        }
    }


def _completed(bench, params, value, ts="2026-08-01T00:00:00", device="rigetti_cepheus-1-108q"):
    return {
        "provider": "aws",
        "device": device,
        "timestamp": ts,
        "job_type": bench,
        "params": {"benchmark_name": bench, **params},
        "results": {"score": value},
        "directions": {"score": "higher"},
    }


def _outcome(bench, params, outcome, ts="2026-08-07T12:00:00", detail=None, device="rigetti_cepheus-1-108q"):
    row = {
        "provider": "aws",
        "device": device,
        "timestamp": ts,
        "job_type": bench,
        "params": {"benchmark_name": bench, **params},
        "outcome": outcome,
        "results": None,
    }
    if detail is not None:
        row["outcome_detail"] = detail
    return row


def _composite(rows):
    row_series = {id(r): SERIES for r in rows}
    baseline_avg = {
        SERIES: {
            (LRQAOA, "score", _selector_fingerprint({"num_qubits": 100})): 0.5,
            (WIT, "score", _selector_fingerprint(None)): 0.5,
        }
    }
    out = compute_device_composite_scores(rows, row_series, baseline_avg, _scoring_cfg())
    return {rec["device"]: rec for rec in out}


class ComponentOutcomeStampingTests(unittest.TestCase):
    def test_outcome_is_stamped_on_missing_component(self):
        detail = {
            "reason": "Compiler rejects 100-qubit LR-QAOA circuits",
            "error_message": "compilation failed: too many qubits in routing",
            "source": "dispatch",
        }
        rows = [
            _completed(WIT, {}, 0.25),
            _outcome(LRQAOA, {"num_qubits": 100}, "unsupported", detail=detail),
        ]
        rec = _composite(rows)["rigetti_cepheus-1-108q"]
        comp = rec["components"]["LR-QAOA 100q"]
        self.assertEqual(comp["reported_outcome"], "unsupported")
        self.assertEqual(comp["reported_outcome_reason"], detail["reason"])
        self.assertEqual(comp["reported_outcome_timestamp"], "2026-08-07T12:00:00")
        self.assertNotIn("reported_outcome_detail", comp)
        # Scoring is unchanged: no value, weight stays in the denominator.
        self.assertFalse(comp["normalized_available"])
        self.assertFalse(comp["raw_available"])
        self.assertEqual(comp["required_num_qubits"], 100)
        self.assertAlmostEqual(rec["metriq_score"], 25.0)
        # Completed components carry no outcome fields.
        self.assertNotIn("reported_outcome", rec["components"]["WIT"])

    def test_completed_record_supersedes_outcome_regardless_of_order(self):
        rows = [
            _completed(LRQAOA, {"num_qubits": 100}, 0.5, ts="2026-08-01T00:00:00"),
            _outcome(LRQAOA, {"num_qubits": 100}, "error", ts="2026-08-09T00:00:00"),
        ]
        comp = _composite(rows)["rigetti_cepheus-1-108q"]["components"]["LR-QAOA 100q"]
        self.assertNotIn("reported_outcome", comp)
        self.assertTrue(comp["raw_available"])

    def test_latest_outcome_wins(self):
        rows = [
            _outcome(LRQAOA, {"num_qubits": 100}, "error", ts="2026-08-01T00:00:00",
                     detail={"reason": "queue died"}),
            _outcome(LRQAOA, {"num_qubits": 100}, "unsupported", ts="2026-08-09T00:00:00",
                     detail={"reason": "compiler limit"}),
            _outcome(LRQAOA, {"num_qubits": 100}, "error", ts="2026-08-05T00:00:00"),
        ]
        comp = _composite(rows)["rigetti_cepheus-1-108q"]["components"]["LR-QAOA 100q"]
        self.assertEqual(comp["reported_outcome"], "unsupported")
        self.assertEqual(comp["reported_outcome_reason"], "compiler limit")
        self.assertEqual(comp["reported_outcome_timestamp"], "2026-08-09T00:00:00")

    def test_outcome_for_other_instance_is_not_stamped(self):
        # A claim about LR-QAOA at 50 qubits says nothing about the 100-qubit component.
        rows = [_outcome(LRQAOA, {"num_qubits": 50}, "unsupported")]
        comp = _composite(rows)["rigetti_cepheus-1-108q"]["components"]["LR-QAOA 100q"]
        self.assertNotIn("reported_outcome", comp)

    def test_outcome_without_detail(self):
        rows = [_outcome(WIT, {}, "not_applicable")]
        comp = _composite(rows)["rigetti_cepheus-1-108q"]["components"]["WIT"]
        self.assertEqual(comp["reported_outcome"], "not_applicable")
        self.assertIsNone(comp["reported_outcome_reason"])
        self.assertEqual(comp["reported_outcome_timestamp"], "2026-08-07T12:00:00")


if __name__ == "__main__":
    unittest.main()
