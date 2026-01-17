#!/usr/bin/env python3
"""
aggregate.py — ETL for metriq-data

Scans all v* result payloads recursively (both `results.json` and per-run `*.json`),
aggregates rows,
and writes outputs to:
  - dist/benchmark.latest.json
  - dist/platforms/index.json
  - dist/platforms/<provider>/<device>.json
"""

import os

import json

from etl import (
    iso_utc_now,
    ensure_dir,
    find_result_files,
    collect_flat_rows_and_registry,
    write_platform_outputs,
    write_benchmark_latest,
)
from score import (
    apply_custom_metric_derivations,
    load_scoring_config,
    validate_scoring_config,
    compute_baseline_averages_by_series,
    compute_and_attach_metriq_scores,
    compute_device_composite_scores,
)


def main(argv: list[str] | None = None) -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    dist_path = os.path.join(root, "dist")
    ensure_dir(dist_path)

    generated_at = iso_utc_now()

    files = find_result_files(root)
    total_files = len(files)
    flat_rows, row_series, registry = collect_flat_rows_and_registry(root, files)
    apply_custom_metric_derivations(flat_rows)

    # Scoring configured only via scripts/scoring.json
    scoring_cfg = load_scoring_config(root)
    # Validate composite weight sums per series/default
    validate_scoring_config(scoring_cfg)

    baseline_avg_by_series, baseline_choice_summary = compute_baseline_averages_by_series(
        flat_rows,
        row_series,
        scoring_cfg,
    )

    compute_and_attach_metriq_scores(flat_rows, row_series, baseline_avg_by_series, scoring_cfg)

    # Composite Metriq Score across benchmarks (used only in platform files)
    composite_records = compute_device_composite_scores(
        flat_rows,
        row_series,
        baseline_avg_by_series,
        scoring_cfg,
    )

    # Also write per-series outputs mirroring source dataset structure
    #  - dist/<series>/benchmark.latest.json
    series_labels = sorted(set(row_series.values()))
    for s in series_labels:
        s_dir = os.path.join(dist_path, s)
        ensure_dir(s_dir)
        # Per-series latest benchmark rows
        s_rows = [r for r in flat_rows if row_series.get(id(r)) == s]
        s_latest = os.path.join(s_dir, "benchmark.latest.json")
        with open(s_latest, "w", encoding="utf-8") as f:
            f.write(json.dumps(s_rows, ensure_ascii=False, indent=2))
            f.write("\n")


    latest_file = write_benchmark_latest(dist_path, flat_rows)
    platform_count, platforms_dir = write_platform_outputs(
        registry,
        dist_path,
        generated_at,
        composite_records,
    )

    print(
        f"Processed {len(flat_rows)} rows across {platform_count} platforms from {total_files} files."
    )
    if baseline_choice_summary:
        print("Baselines:")
        for line in baseline_choice_summary:
            print(f"  - {line}")
    wrote = [os.path.relpath(latest_file, root), f"{os.path.relpath(platforms_dir, root)}/"]
    for s in series_labels:
        wrote.append(f"dist/{s}/benchmark.latest.json")
    print(f"Wrote: {', '.join(wrote)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
