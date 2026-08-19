#!/usr/bin/env python3
"""Assemble an isolated data-PR preview from trusted production web assets."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


VERSION_DIR_RE = re.compile(r"v\d+(?:\.\d+)*")
PATH_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SHA_RE = re.compile(r"[0-9a-f]{40}")
UNSAFE_URI_RE = re.compile(r"\s*(?:data|javascript|vbscript):", re.IGNORECASE)
CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x1f\x7f]")
HTTPS_URL_RE = re.compile(r"https://[^\s]+", re.IGNORECASE)


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def is_allowed_data_path(relative_path: Path) -> bool:
    parts = relative_path.parts
    if parts == ("benchmark.latest.json",):
        return True
    if parts == ("platforms", "index.json"):
        return True
    if (
        len(parts) == 3
        and parts[0] == "platforms"
        and PATH_SEGMENT_RE.fullmatch(parts[1]) is not None
        and Path(parts[2]).suffix == ".json"
        and PATH_SEGMENT_RE.fullmatch(Path(parts[2]).stem) is not None
    ):
        return True
    return (
        len(parts) == 2
        and VERSION_DIR_RE.fullmatch(parts[0]) is not None
        and parts[1] == "benchmark.latest.json"
    )


def is_allowed_data_directory(relative_path: Path) -> bool:
    parts = relative_path.parts
    if parts == ("platforms",):
        return True
    if len(parts) == 1 and VERSION_DIR_RE.fullmatch(parts[0]) is not None:
        return True
    return (
        len(parts) == 2
        and parts[0] == "platforms"
        and PATH_SEGMENT_RE.fullmatch(parts[1]) is not None
    )


def _allows_outcome_error_angle_brackets(
    relative_path: Path, field_path: tuple[str | None, ...]
) -> bool:
    parts = relative_path.parts
    is_benchmark_file = parts == ("benchmark.latest.json",) or (
        len(parts) == 2
        and VERSION_DIR_RE.fullmatch(parts[0]) is not None
        and parts[1] == "benchmark.latest.json"
    )
    return is_benchmark_file and field_path == (
        None,
        "outcome_detail",
        "error_message",
    )


def validate_json_content(value: object, relative_path: Path) -> None:
    pending: list[tuple[object, str | None, tuple[str | None, ...]]] = [
        (value, None, ())
    ]
    while pending:
        item, field_name, field_path = pending.pop()
        if isinstance(item, dict):
            pending.extend((key, None, ()) for key in item)
            pending.extend(
                (child, key.lower(), (*field_path, key))
                for key, child in item.items()
            )
        elif isinstance(item, list):
            pending.extend(
                (child, field_name, (*field_path, None)) for child in item
            )
        elif isinstance(item, str):
            has_angle_brackets = "<" in item or ">" in item
            if (
                has_angle_brackets
                and not _allows_outcome_error_angle_brackets(
                    relative_path, field_path
                )
            ) or CONTROL_CHARACTER_RE.search(item):
                raise ValueError(f"unsafe string in preview JSON: {relative_path}")
            if UNSAFE_URI_RE.match(item):
                raise ValueError(f"unsafe URI in preview JSON: {relative_path}")
            if field_name and field_name.endswith(("url", "uri")) and item:
                if HTTPS_URL_RE.fullmatch(item) is None:
                    raise ValueError(f"non-HTTPS URL in preview JSON: {relative_path}")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"non-finite number in preview JSON: {relative_path}")


def validate_data_tree(data_directory: Path) -> None:
    required = (
        data_directory / "benchmark.latest.json",
        data_directory / "platforms" / "index.json",
    )
    for path in required:
        if not path.is_file():
            raise ValueError(f"missing required preview data file: {path}")

    for path in data_directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"preview data must not contain symlinks: {path}")
        relative_path = path.relative_to(data_directory)
        if path.is_dir():
            if not is_allowed_data_directory(relative_path):
                raise ValueError(f"unexpected preview data directory: {relative_path}")
            continue
        if not path.is_file():
            raise ValueError(f"unexpected preview data entry: {relative_path}")
        if not is_allowed_data_path(relative_path):
            raise ValueError(f"unexpected preview data path: {relative_path}")
        try:
            with path.open(encoding="utf-8") as stream:
                value = json.load(stream)
            validate_json_content(value, relative_path)
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError(f"invalid preview JSON: {relative_path}") from error


def preview_banner(pr_number: int) -> str:
    pull_request_url = (
        f"https://github.com/unitaryfoundation/metriq-data/pull/{pr_number}"
    )
    return f"""
    <div role="status" style="background:#e0f2fe;border-bottom:1px solid #7dd3fc;color:#0c4a6e;display:flex;flex-wrap:wrap;font:600 15px/1.5 system-ui,sans-serif;gap:8px 20px;justify-content:center;padding:12px 20px;text-align:center">
      <span><strong>Staging preview:</strong> production UI with the merge result from <a href="{pull_request_url}" style="color:#075985" target="_blank" rel="noopener noreferrer">metriq-data PR #{pr_number}</a>. These scores are not live.</span>
      <a href="https://metriq.info/" style="color:#075985">Open production data</a>
    </div>
    """.strip()


def prepare_preview(
    web_directory: Path,
    data_directory: Path,
    output_directory: Path,
    pr_number: int,
    workflow_sha: str,
) -> None:
    web_directory = web_directory.resolve()
    data_directory = data_directory.resolve()
    output_directory = output_directory.resolve()

    if not web_directory.is_dir():
        raise ValueError(f"production web directory does not exist: {web_directory}")
    if not data_directory.is_dir():
        raise ValueError(f"preview data directory does not exist: {data_directory}")
    if output_directory.exists():
        raise ValueError(f"preview output already exists: {output_directory}")
    if pr_number <= 0:
        raise ValueError("pull request number must be positive")
    if SHA_RE.fullmatch(workflow_sha) is None:
        raise ValueError("workflow SHA must be a 40-character lowercase hex digest")

    validate_data_tree(data_directory)

    shutil.copytree(
        web_directory,
        output_directory,
        ignore=shutil.ignore_patterns(".git", "CNAME", "pr-preview"),
    )
    preview_data_directory = output_directory / "metriq-data"
    shutil.copytree(data_directory, preview_data_directory)

    config_path = output_directory / "data" / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("production web bundle has no valid data/config.json") from error
    if not isinstance(config, dict):
        raise ValueError("production web data/config.json must contain an object")
    config.update(
        {
            "benchmarksUrl": "./metriq-data/benchmark.latest.json",
            "platformsIndexUrl": "./metriq-data/platforms/index.json",
        }
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    index_path = output_directory / "index.html"
    try:
        index_html = index_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError("production web bundle has no readable index.html") from error
    if index_html.count("<head>") != 1 or index_html.count("<body>") != 1:
        raise ValueError("production web index.html has an unsupported document structure")
    index_html = index_html.replace(
        "<head>",
        '<head>\n    <meta name="robots" content="noindex, nofollow" />',
        1,
    )
    index_html = index_html.replace("<body>", f"<body>\n    {preview_banner(pr_number)}", 1)
    index_path.write_text(index_html, encoding="utf-8")

    manifest = {
        "pull_request": pr_number,
        "workflow_sha": workflow_sha,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "web_source": "unitaryfoundation/metriq-web@gh-pages",
    }
    (preview_data_directory / "preview-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pr-number", type=positive_integer, required=True)
    parser.add_argument("--workflow-sha", required=True)
    args = parser.parse_args()
    prepare_preview(
        args.web_dir,
        args.data_dir,
        args.output_dir,
        args.pr_number,
        args.workflow_sha,
    )


if __name__ == "__main__":
    main()
