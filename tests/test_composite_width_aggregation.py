import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from score import (  # noqa: E402
    _selector_fingerprint,
    compute_device_composite_scores,
)


BENCH = "Mirror Circuits"
SELECTORS = [{"width": 8}, {"width": 64}]
SUB_WEIGHTS = ["1/9", "8/9"]


def _scoring_cfg(aggregation=None, weight="1"):
    group = {
        "label": BENCH,
        "weight": weight,
        "components": [
            {
                "benchmark": BENCH,
                "metric": "score",
                "selector": sel,
                "label": f"{BENCH} (w={sel['width']}):score",
                "weight": sub,
            }
            for sel, sub in zip(SELECTORS, SUB_WEIGHTS)
        ],
    }
    if aggregation:
        group["aggregation"] = aggregation
    return {
        "default": {
            "baseline": {"provider": "ibm", "device": "base_device"},
            "composite": {"components": [group]},
        }
    }


def _row(device, selector, value, direction="higher"):
    return {
        "provider": "ibm",
        "device": device,
        "timestamp": "2026-01-01T00:00:00",
        "job_type": BENCH,
        "params": dict(selector),
        "results": {"score": value},
        "directions": {"score": direction},
    }


def _score(rows, baseline_values, scoring_cfg=None, series="v1.0"):
    """Run the composite over rows, returning {device: metriq_score}."""
    scoring_cfg = scoring_cfg or _scoring_cfg()
    row_series = {id(r): series for r in rows}
    baseline_avg = {
        series: {
            (BENCH, "score", _selector_fingerprint(sel)): val
            for sel, val in zip(SELECTORS, baseline_values)
        }
    }
    out = compute_device_composite_scores(rows, row_series, baseline_avg, scoring_cfg)
    return {rec["device"]: rec for rec in out}


class WidthAggregationTests(unittest.TestCase):
    def test_near_zero_baseline_width_does_not_dominate(self):
        """A width whose baseline sits at the noise floor must not swamp the group.

        Mirrors the real ibm_torino/ibm_boston case: the baseline scored 0.0037 at
        w=64, so normalizing that width on its own yields a 7192% ratio that, under
        per-width normalization, supplied over half the device's composite.
        """
        rows = [
            _row("dev", SELECTORS[0], 0.7477),
            _row("dev", SELECTORS[1], 0.2661),
        ]
        recs = _score(rows, [0.3172, 0.0037])

        # Width-aggregated: (1/9 * 0.7477 + 8/9 * 0.2661) / (1/9 * 0.3172 + 8/9 * 0.0037)
        expected = 100.0 * (
            (0.7477 / 9 + 8 * 0.2661 / 9) / (0.3172 / 9 + 8 * 0.0037 / 9)
        )
        self.assertAlmostEqual(recs["dev"]["metriq_score"], expected, places=6)

        # Per-width normalization would have averaged 235.7% and 7191.9%, landing
        # an order of magnitude higher.
        per_width = (100.0 * 0.7477 / 0.3172) / 9 + 8 * (100.0 * 0.2661 / 0.0037) / 9
        self.assertLess(recs["dev"]["metriq_score"], per_width / 5)

    def test_group_subscore_is_shared_by_every_width(self):
        rows = [
            _row("dev", SELECTORS[0], 0.7477),
            _row("dev", SELECTORS[1], 0.2661),
        ]
        rec = _score(rows, [0.3172, 0.0037])["dev"]
        subscores = {c["group_subscore"] for c in rec["components"].values()}
        self.assertEqual(len(subscores), 1)
        # The group carries the full composite weight here, so the shared subscore
        # is the device's score.
        self.assertAlmostEqual(subscores.pop(), rec["metriq_score"], places=6)

    def test_baseline_device_scores_exactly_100(self):
        rows = [
            _row("base_device", SELECTORS[0], 0.3172),
            _row("base_device", SELECTORS[1], 0.0037),
        ]
        recs = _score(rows, [0.3172, 0.0037])
        self.assertAlmostEqual(recs["base_device"]["metriq_score"], 100.0, places=9)

    def test_baseline_device_anchors_even_when_a_width_measured_zero(self):
        """A measured 0.0 is a result, not a missing submission.

        The baseline cannot normalize its own zero, but the width is still present,
        so it must not draw a coverage penalty.
        """
        rows = [
            _row("base_device", SELECTORS[0], 0.3172),
            _row("base_device", SELECTORS[1], 0.0),
        ]
        recs = _score(rows, [0.3172, 0.0])
        self.assertAlmostEqual(recs["base_device"]["metriq_score"], 100.0, places=9)

    def test_missing_width_is_penalized_by_its_sub_weight(self):
        rows = [_row("dev", SELECTORS[0], 0.3172)]
        recs = _score(rows, [0.3172, 0.0037])
        # Parity on the only submitted width, which carries 1/9 of the group.
        self.assertAlmostEqual(recs["dev"]["metriq_score"], 100.0 / 9, places=6)

    def test_lower_is_better_inverts_the_aggregate_ratio(self):
        rows = [
            _row("dev", SELECTORS[0], 0.002, direction="lower"),
            _row("dev", SELECTORS[1], 0.007, direction="lower"),
        ]
        recs = _score(rows, [0.005, 0.068])
        expected = 100.0 * (
            (0.005 / 9 + 8 * 0.068 / 9) / (0.002 / 9 + 8 * 0.007 / 9)
        )
        self.assertAlmostEqual(recs["dev"]["metriq_score"], expected, places=6)

    def test_falls_back_to_normalized_scores_when_raw_values_are_absent(self):
        """Derived metrics such as BSEQ carry no raw value to aggregate."""
        rows = [
            {
                "provider": "ibm",
                "device": "dev",
                "timestamp": "2026-01-01T00:00:00",
                "job_type": BENCH,
                "params": dict(SELECTORS[0]),
                "normalized_scores": {"score": 150.0},
            },
            {
                "provider": "ibm",
                "device": "dev",
                "timestamp": "2026-01-01T00:00:00",
                "job_type": BENCH,
                "params": dict(SELECTORS[1]),
                "normalized_scores": {"score": 300.0},
            },
        ]
        recs = _score(rows, [0.3172, 0.0037])
        self.assertAlmostEqual(
            recs["dev"]["metriq_score"], 150.0 / 9 + 8 * 300.0 / 9, places=6
        )

    def test_harmonic_groups_are_unchanged(self):
        rows = [
            _row("dev", SELECTORS[0], 0.002, direction="lower"),
            _row("dev", SELECTORS[1], 0.007, direction="lower"),
        ]
        recs = _score(rows, [0.005, 0.068], scoring_cfg=_scoring_cfg(aggregation="harmonic"))
        # Weighted harmonic mean of the per-width normalized values.
        n1 = 100.0 * 0.005 / 0.002
        n2 = 100.0 * 0.068 / 0.007
        expected = 1.0 / ((1.0 / 9) / n1 + (8.0 / 9) / n2)
        self.assertAlmostEqual(recs["dev"]["metriq_score"], expected, places=6)


if __name__ == "__main__":
    unittest.main()
