import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_preview import (  # noqa: E402
    is_allowed_data_directory,
    is_allowed_data_path,
    prepare_preview,
    validate_data_tree,
)


class PreparePreviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.web = self.root / "web"
        self.data = self.root / "data"
        self.output = self.root / "output"

        (self.web / "data").mkdir(parents=True)
        (self.web / "data" / "config.json").write_text(
            json.dumps(
                {
                    "benchmarksUrl": "https://example.com/production.json",
                    "platformsIndexUrl": "https://example.com/platforms.json",
                    "hiddenProviders": ["local"],
                }
            ),
            encoding="utf-8",
        )
        (self.web / "index.html").write_text(
            "<!doctype html><html><head><title>Metriq</title></head><body><main>Site</main></body></html>",
            encoding="utf-8",
        )
        (self.web / "main.js").write_text("console.log('production');\n", encoding="utf-8")
        (self.web / "CNAME").write_text("metriq.info\n", encoding="utf-8")
        (self.web / ".git").mkdir()
        (self.web / ".git" / "config").write_text("ignored\n", encoding="utf-8")
        (self.web / "pr-preview" / "pr-1").mkdir(parents=True)
        (self.web / "pr-preview" / "pr-1" / "index.html").write_text(
            "unrelated web preview\n", encoding="utf-8"
        )

        (self.data / "platforms" / "ibm").mkdir(parents=True)
        (self.data / "v0.7").mkdir()
        (self.data / "benchmark.latest.json").write_text("[]\n", encoding="utf-8")
        (self.data / "platforms" / "index.json").write_text(
            '{"platforms": []}\n', encoding="utf-8"
        )
        (self.data / "platforms" / "ibm" / "ibm_fez.json").write_text(
            '{}\n', encoding="utf-8"
        )
        (self.data / "v0.7" / "benchmark.latest.json").write_text(
            "[]\n", encoding="utf-8"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_assembles_production_ui_with_preview_data_and_notice(self):
        prepare_preview(
            self.web,
            self.data,
            self.output,
            pr_number=514,
            workflow_sha="a" * 40,
        )

        config = json.loads((self.output / "data" / "config.json").read_text())
        self.assertEqual(config["benchmarksUrl"], "./metriq-data/benchmark.latest.json")
        self.assertEqual(
            config["platformsIndexUrl"], "./metriq-data/platforms/index.json"
        )
        self.assertEqual(config["hiddenProviders"], ["local"])

        html = (self.output / "index.html").read_text()
        self.assertIn("Staging preview:", html)
        self.assertIn("metriq-data PR #514", html)
        self.assertIn('name="robots" content="noindex, nofollow"', html)
        self.assertTrue((self.output / "main.js").is_file())
        self.assertTrue(
            (self.output / "metriq-data" / "platforms" / "ibm" / "ibm_fez.json").is_file()
        )
        self.assertFalse((self.output / "CNAME").exists())
        self.assertFalse((self.output / ".git").exists())
        self.assertFalse((self.output / "pr-preview").exists())

        manifest = json.loads(
            (self.output / "metriq-data" / "preview-manifest.json").read_text()
        )
        self.assertEqual(manifest["pull_request"], 514)
        self.assertEqual(manifest["workflow_sha"], "a" * 40)

    def test_rejects_unexpected_json_paths(self):
        (self.data / "attacker.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unexpected preview data path"):
            validate_data_tree(self.data)

    def test_rejects_non_json_files(self):
        (self.data / "index.html").write_text("<script></script>", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unexpected preview data path"):
            validate_data_tree(self.data)

    def test_rejects_symlinks(self):
        symlink = self.data / "platforms" / "ibm" / "linked.json"
        try:
            symlink.symlink_to(self.data / "benchmark.latest.json")
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"symlink creation is unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
            validate_data_tree(self.data)

    def test_rejects_html_and_executable_urls_in_json_values(self):
        detail = self.data / "platforms" / "ibm" / "ibm_fez.json"
        for payload in (
            {"first_seen": '<img src=x onerror="alert(1)">'},
            [{"outcome_detail": {"error_message": "expected <T>"}}],
            {"lifecycle": {"source_url": "javascript:alert(1)"}},
            {"lifecycle": {"source_url": "http://example.com/source"}},
            {"label": "line\nbreak"},
        ):
            detail.write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(payload=payload), self.assertRaisesRegex(
                ValueError, "unsafe|non-HTTPS"
            ):
                validate_data_tree(self.data)

    def test_allows_angle_brackets_in_outcome_error_messages(self):
        payload = [
            {
                "outcome": "error",
                "outcome_detail": {
                    "error_message": "expected Foo<Bar>, got Baz<Qux>"
                },
            }
        ]
        for path in (
            self.data / "benchmark.latest.json",
            self.data / "v0.7" / "benchmark.latest.json",
        ):
            path.write_text(json.dumps(payload), encoding="utf-8")

        validate_data_tree(self.data)

    def test_rejects_angle_brackets_outside_outcome_error_messages(self):
        benchmark = self.data / "benchmark.latest.json"
        for payload in (
            [{"error_message": "expected <T>"}],
            [{"outcome_detail": {"reason": "expected <T>"}}],
            [
                {
                    "wrapper": {
                        "outcome_detail": {"error_message": "expected <T>"}
                    }
                }
            ],
            [{"outcome_detail": {"error_message": ["expected <T>"]}}],
        ):
            benchmark.write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(payload=payload), self.assertRaisesRegex(
                ValueError, "unsafe string"
            ):
                validate_data_tree(self.data)

    def test_keeps_other_validation_for_outcome_error_messages(self):
        benchmark = self.data / "benchmark.latest.json"
        for error_message in ("line\nbreak", "javascript:alert(1)"):
            payload = [
                {"outcome_detail": {"error_message": error_message}}
            ]
            benchmark.write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(error_message=error_message), self.assertRaisesRegex(
                ValueError, "unsafe"
            ):
                validate_data_tree(self.data)

    def test_rejects_unexpected_directories(self):
        (self.data / "unpublished" / "nested").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "unexpected preview data directory"):
            validate_data_tree(self.data)

    def test_allowlist_covers_only_web_consumed_outputs(self):
        for path in (
            "benchmark.latest.json",
            "v0.7/benchmark.latest.json",
            "v1.2.3/benchmark.latest.json",
            "platforms/index.json",
            "platforms/ibm/ibm_fez.json",
        ):
            self.assertTrue(is_allowed_data_path(Path(path)), path)
        for path in (
            "preview-manifest.json",
            "v0.7/other.json",
            "platforms/index.html",
            "platforms/ibm/nested/device.json",
            "other/file.json",
        ):
            self.assertFalse(is_allowed_data_path(Path(path)), path)

        for path in ("platforms", "platforms/ibm", "v0.7"):
            self.assertTrue(is_allowed_data_directory(Path(path)), path)
        for path in ("other", "platforms/ibm/nested", "v-next"):
            self.assertFalse(is_allowed_data_directory(Path(path)), path)


if __name__ == "__main__":
    unittest.main()
