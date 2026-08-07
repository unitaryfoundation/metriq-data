import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from etl import flatten_row  # noqa: E402


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

    def test_unknown_outcome_values_are_dropped(self):
        out = flatten_row(outcome_record(outcome="exploded"))
        self.assertNotIn("outcome", out)
        self.assertNotIn("outcome_detail", out)

    def test_outcome_detail_is_normalized(self):
        out = flatten_row(
            outcome_record(
                outcome="  ERROR ",
                outcome_detail={"reason": "  queue died  ", "unknown_key": "x", "source": ""},
            )
        )
        self.assertEqual(out["outcome"], "error")
        self.assertEqual(out["outcome_detail"], {"reason": "queue died"})

    def test_outcome_without_detail(self):
        out = flatten_row(outcome_record(outcome_detail=None))
        self.assertEqual(out["outcome"], "unsupported")
        self.assertNotIn("outcome_detail", out)


if __name__ == "__main__":
    unittest.main()
