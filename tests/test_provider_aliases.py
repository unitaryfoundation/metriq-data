import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from etl import (  # noqa: E402
    canonical_device_name,
    canonical_provider_name,
    flatten_row,
    load_platform_catalog,
    upsert_platform,
    write_platform_outputs,
)
from score import (  # noqa: E402
    _baseline_provider_device_for_series,
    baseline_metadata_for_latest_series,
)


CEPHEUS_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"


class ProviderAliasTests(unittest.TestCase):
    def test_aws_and_braket_share_canonical_platform_identity(self):
        for alias in ("aws", "AWS", "braket", "Braket"):
            with self.subTest(alias=alias):
                self.assertEqual(canonical_provider_name(alias), "aws")
                self.assertEqual(
                    canonical_device_name(alias, CEPHEUS_ARN),
                    "rigetti_cepheus-1-108q",
                )

    def test_flatten_and_registry_canonicalize_legacy_braket_record(self):
        row = {
            "timestamp": "2026-07-15T16:26:08",
            "platform": {
                "provider": "braket",
                "device": CEPHEUS_ARN,
                "device_metadata": {"num_qubits": 107},
            },
            "results": {"score": {"value": 0.5}},
        }

        flat = flatten_row(row)
        self.assertEqual(flat["provider"], "aws")
        self.assertEqual(flat["device"], "rigetti_cepheus-1-108q")

        registry = {}
        upsert_platform(registry, row, "result.json", "v0.7")
        self.assertIn(("aws", "rigetti_cepheus-1-108q"), registry)

        with tempfile.TemporaryDirectory() as tmp:
            count, _ = write_platform_outputs(registry, tmp, "2026-07-16T00:00:00Z")
            self.assertEqual(count, 1)
            self.assertTrue(
                (Path(tmp) / "platforms" / "aws" / "rigetti_cepheus-1-108q.json").is_file()
            )

    def test_catalog_and_scoring_config_accept_braket_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            scripts_dir = Path(tmp) / "scripts"
            scripts_dir.mkdir()
            (scripts_dir / "platform_catalog.json").write_text(
                json.dumps(
                    {
                        "platforms": [
                            {
                                "provider": "braket",
                                "device": CEPHEUS_ARN,
                                "lifecycle": {"status": "active"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            catalog = load_platform_catalog(tmp)

        self.assertEqual(
            catalog[("aws", "rigetti_cepheus-1-108q")],
            {"lifecycle": {"status": "active"}},
        )

        config = {"default": {"baseline": {"provider": "braket", "device": CEPHEUS_ARN}}}
        self.assertEqual(
            _baseline_provider_device_for_series(config, "v0.7"),
            ("aws", "rigetti_cepheus-1-108q"),
        )
        self.assertEqual(
            baseline_metadata_for_latest_series(config, ["v0.9", "v0.10", "unknown"]),
            {
                "provider": "aws",
                "device": "rigetti_cepheus-1-108q",
                "series": "v0.10",
            },
        )

    def test_platform_index_publishes_baseline_metadata(self):
        row = {
            "timestamp": "2026-07-15T16:26:08",
            "platform": {"provider": "ibm", "device": "ibm_boston"},
            "results": {"score": {"value": 0.5}},
        }
        registry = {}
        upsert_platform(registry, row, "result.json", "v0.7")
        baseline = {"provider": "ibm", "device": "ibm_boston", "series": "v0.7"}

        with tempfile.TemporaryDirectory() as tmp:
            write_platform_outputs(
                registry,
                tmp,
                "2026-07-16T00:00:00Z",
                baseline=baseline,
            )
            index = json.loads(
                (Path(tmp) / "platforms" / "index.json").read_text(encoding="utf-8")
            )

        self.assertEqual(index["baseline"], baseline)


if __name__ == "__main__":
    unittest.main()
