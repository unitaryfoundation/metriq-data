from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

from etl import derive_benchmark_name


def load_baselines_config(root: str) -> dict[str, Any]:
    """Load baseline configuration from scripts/baselines.json if present.

    Expected shape:
      {
        "series": {
          "v0.4": { "provider": "ibm", "device": "ibm_torino" }
        },
        "default": { "provider": "ibm", "device": "ibm_torino" }
      }
    """
    cfg_path = os.path.join(root, "scripts", "baselines.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Warning: failed to load baselines config: {e}", file=sys.stderr)
    return {}


def compute_baseline_averages_by_series(
    flat_rows: list[dict[str, Any]],
    row_series: dict[int, str],
    baselines_cfg: dict[str, Any],
) -> tuple[dict[str, dict[tuple[str, str], float]], list[str]]:
    series_list = sorted(set(row_series.values()))
    baseline_avg_by_series: dict[str, dict[tuple[str, str], float]] = {}
    summary: list[str] = []

    for series in series_list:
        series_cfg = (baselines_cfg.get("series", {}) or {}).get(series, {}) if isinstance(baselines_cfg, dict) else {}
        base_provider = series_cfg.get("provider") or (baselines_cfg.get("default", {}) or {}).get("provider")
        base_device = series_cfg.get("device") or (baselines_cfg.get("default", {}) or {}).get("device")

        if not base_provider or not base_device:
            baseline_avg_by_series[series] = {}
            summary.append(f"{series}: (no baseline configured)")
            continue

        selected = [
            r for r in flat_rows
            if r.get("provider") == base_provider and r.get("device") == base_device and row_series.get(id(r)) == series
        ]

        baseline_values: dict[tuple[str, str], list[float]] = {}
        for r in selected:
            results = r.get("results") if isinstance(r.get("results"), dict) else {}
            bench = derive_benchmark_name(r)
            for metric, val in results.items():
                try:
                    num = float(val)
                except Exception:
                    continue
                if not (num == num):  # NaN
                    continue
                baseline_values.setdefault((bench, metric), []).append(num)

        baseline_avg: dict[tuple[str, str], float] = {}
        for key, vals in baseline_values.items():
            if not vals:
                continue
            try:
                baseline_avg[key] = sum(vals) / len(vals)
            except Exception:
                pass
        baseline_avg_by_series[series] = baseline_avg
        summary.append(f"{series}: {base_provider}/{base_device}")

    return baseline_avg_by_series, summary


def compute_and_attach_metriq_scores(
    flat_rows: list[dict[str, Any]],
    row_series: dict[int, str],
    baseline_avg_by_series: dict[str, dict[tuple[str, str], float]],
) -> None:
    for r in flat_rows:
        results = r.get("results") if isinstance(r.get("results"), dict) else {}
        if not results:
            continue
        bench = derive_benchmark_name(r)
        dir_map = r.get("directions") if isinstance(r.get("directions"), dict) else {}
        series = row_series.get(id(r))
        baseline_avg = baseline_avg_by_series.get(series or "", {})
        scores: dict[str, float] = {}
        for metric, val in results.items():
            try:
                v = float(val)
            except Exception:
                continue
            if not (v == v):  # NaN
                continue
            base = baseline_avg.get((bench, metric))
            if base is None:
                continue
            direction = str(dir_map.get(metric, "higher")).lower()
            score: Optional[float] = None
            try:
                if direction == "lower":
                    if v > 0:
                        score = (base / v) * 100.0
                else:
                    if base > 0:
                        score = (v / base) * 100.0
            except Exception:
                score = None
            if score is not None and (score == score) and score not in (float("inf"), float("-inf")):
                scores[metric] = score
        if not scores:
            continue

        # If the benchmark reports a single metric and we computed exactly one score,
        # expose a single scalar `metriq_score` instead of a per-metric mapping.
        # This avoids nesting metric names as keys of `metriq_score` for single-metric benchmarks
        # (e.g., WIT).
        if len(scores) == 1 and len(results) == 1:
            # Extract the only score value
            r["metriq_score"] = next(iter(scores.values()))
        else:
            # For multi-metric benchmarks we currently do not publish a scalar until
            # a composite scoring spec is defined. Leave `metriq_score` unset.
            # (We keep the per-metric scores internal in case a downstream composite
            # is added later.)
            pass
