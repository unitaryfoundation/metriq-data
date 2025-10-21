#!/usr/bin/env python3
"""
aggregate.py — ETL for metriq-data

Scans all v*/<provider>/results.json files recursively, aggregates rows,
and writes outputs to:
  - dist/benchmark.latest.json
  - dist/platforms/index.json
  - dist/platforms/<provider>/<device>.json
"""

import os

from etl import (
    iso_utc_now,
    ensure_dir,
    find_result_files,
    collect_flat_rows_and_registry,
    write_platform_outputs,
    write_benchmark_latest,
)
from score import (
    load_baselines_config,
    compute_baseline_averages_by_series,
    compute_and_attach_metriq_scores,
)


def main(argv: list[str] | None = None) -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    dist_path = os.path.join(root, "dist")
    ensure_dir(dist_path)

    generated_at = iso_utc_now()

    files = find_result_files(root)
    total_files = len(files)
    flat_rows, row_series, registry = collect_flat_rows_and_registry(root, files)

    # Baseline configured only via scripts/baselines.json
    baselines_cfg = load_baselines_config(root)

    baseline_avg_by_series, baseline_choice_summary = compute_baseline_averages_by_series(
        flat_rows,
        row_series,
        baselines_cfg,
    )

    compute_and_attach_metriq_scores(flat_rows, row_series, baseline_avg_by_series)

    latest_file = write_benchmark_latest(dist_path, flat_rows)
    platform_count, platforms_dir = write_platform_outputs(registry, dist_path, generated_at)

    print(
        f"Processed {len(flat_rows)} rows across {platform_count} platforms from {total_files} files."
    )
    if baseline_choice_summary:
        print("Baselines:")
        for line in baseline_choice_summary:
            print(f"  - {line}")
    print(f"Wrote: {os.path.relpath(latest_file, root)} and {os.path.relpath(platforms_dir, root)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

