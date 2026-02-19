from datetime import datetime
import json
import os
import sys
from typing import Any

from etl import canonical_json, derive_benchmark_name, parse_timestamp

def _coerce_float(val: Any) -> float | None:
    try:
        out = float(val)
    except Exception:
        return None
    if not (out == out):
        return None
    return out


def _parse_weight(val: Any) -> float:
    """Parse a weight which may be numeric or a string fraction like '1/7'."""
    if isinstance(val, bool) or val is None:
        raise ValueError(f"Invalid weight: {val!r}")
    if isinstance(val, (int, float)):
        out = float(val)
    elif isinstance(val, str):
        s = val.strip()
        if "/" in s:
            num_s, den_s = (part.strip() for part in s.split("/", 1))
            num = float(num_s)
            den = float(den_s)
            if den == 0:
                raise ValueError(f"Invalid weight (division by zero): {val!r}")
            out = num / den
        else:
            out = float(s)
    else:
        raise ValueError(f"Invalid weight type: {type(val).__name__}")
    if not (out == out) or out in (float("inf"), float("-inf")):
        raise ValueError(f"Invalid weight value: {val!r}")
    return out


def _parse_series_label(label: str | None) -> tuple[int, ...] | None:
    if not isinstance(label, str) or not label.startswith("v"):
        return None
    rest = label[1:]
    if not rest:
        return None
    parts = rest.split(".")
    out: list[int] = []
    for p in parts:
        if not p.isdigit():
            return None
        out.append(int(p))
    return tuple(out)


def _series_major(series_label: str | None) -> int | None:
    """Return semantic major version from a series label like v0.6.1 -> 0."""
    parsed = _parse_series_label(series_label)
    if not parsed:
        return None
    return parsed[0]


def _fallback_baseline_average(
    series_label: str | None,
    bench: str,
    metric: str,
    selector_fp: str,
    baseline_avg_by_series: dict[str, dict[tuple[str, str, str], float]],
) -> float | None:
    """Fallback to same-major latest baseline, then latest earlier series baseline."""
    cur = _parse_series_label(series_label)
    if cur is None:
        return None
    cur_major = cur[0]
    same_major_best_ver: tuple[int, ...] | None = None
    same_major_best_val: float | None = None
    best_ver: tuple[int, ...] | None = None
    best_val: float | None = None
    for s, avg_map in baseline_avg_by_series.items():
        ver = _parse_series_label(s)
        if ver is None:
            continue
        val = avg_map.get((bench, metric, selector_fp))
        if val is None:
            continue
        if ver and ver[0] == cur_major:
            if same_major_best_ver is None or ver > same_major_best_ver:
                same_major_best_ver = ver
                same_major_best_val = val
            continue
        if ver >= cur:
            continue
        if best_ver is None or ver > best_ver:
            best_ver = ver
            best_val = val
    if same_major_best_val is not None:
        return same_major_best_val
    return best_val


def apply_custom_metric_derivations(rows: list[dict[str, Any]]) -> None:
    """Mutate rows in-place to add derived metrics where we define how to aggregate components.

    For benchmarks that report multiple raw components, we can define a scalar metric here
    so downstream metriq-score calculation is well-defined.
    """
    for row in rows:
        bench = derive_benchmark_name(row)
        if bench == "BSEQ":
            _apply_bseq_metric(row)


def _apply_bseq_metric(row: dict[str, Any]) -> None:
    """Legacy hook for BSEQ derived metrics.

    BSEQ scoring is now configured via scripts/scoring.json using baseline-normalized
    component metrics (e.g., largest_connected_size and fraction_connected). We keep
    this function to avoid breaking the derivation pipeline, but do not emit a raw
    `bseq_score` value here (since it depends on baseline normalization).
    """
    return


def _derived_from_components(comp: dict[str, Any]) -> list[dict[str, Any]]:
    items = comp.get("derived_from")
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def _flatten_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten grouped composite components.

    Supports two shapes:
      - Leaf component: {benchmark, metric, weight, ...}
      - Group component: {label, weight, components: [leaf, ...]}

    Returns leaf component dicts augmented with:
      - _group_label, _group_weight, _sub_weight, _effective_weight
    """
    flat: list[dict[str, Any]] = []
    for comp in components:
        if not isinstance(comp, dict):
            continue
        group_children = comp.get("components")
        if isinstance(group_children, list):
            group_label = comp.get("label") or comp.get("benchmark") or "group"
            group_weight = _parse_weight(comp.get("weight", 0.0))
            for child in group_children:
                if not isinstance(child, dict):
                    continue
                sub_weight = _parse_weight(child.get("weight", 0.0))
                merged = dict(child)
                merged["_group_label"] = group_label
                merged["_group_weight"] = group_weight
                merged["_sub_weight"] = sub_weight
                merged["_effective_weight"] = group_weight * sub_weight
                flat.append(merged)
            continue

        sub_weight = _parse_weight(comp.get("weight", 0.0))
        merged = dict(comp)
        merged["_group_label"] = None
        merged["_group_weight"] = 1.0
        merged["_sub_weight"] = sub_weight
        merged["_effective_weight"] = sub_weight
        flat.append(merged)

    return flat


def _compute_derived_normalized_score(
    row: dict[str, Any],
    series_label: str | None,
    selector_fp: str,
    comp: dict[str, Any],
    baseline_avg_by_series: dict[str, dict[tuple[str, str, str], float]],
) -> float | None:
    parts = _derived_from_components(comp)
    if not parts:
        return None
    numerator = 0.0
    denom = 0.0
    for part in parts:
        metric = part.get("metric")
        if not isinstance(metric, str) or not metric:
            continue
        try:
            weight = _parse_weight(part.get("weight", 0.0))
        except Exception:
            continue
        if weight < 0:
            continue
        denom += weight
        v = _get_normalized_metric_value(
            row, metric, series_label, selector_fp, baseline_avg_by_series
        )
        numerator += weight * (v if v is not None else 0.0)
    if denom <= 0:
        return None
    return numerator / denom


def _component_matches_benchmark(comp: dict[str, Any], bench: str) -> bool:
    bench_field = comp.get("benchmark")
    if isinstance(bench_field, str) and bench_field == bench:
        return True
    if isinstance(bench_field, list) and any(isinstance(b, str) and b == bench for b in bench_field):
        return True
    aliases = comp.get("aliases")
    if isinstance(aliases, list) and any(isinstance(a, str) and a == bench for a in aliases):
        return True
    return False


def _selector_fingerprint(selector: dict[str, Any] | None) -> str:
    if not selector:
        return "null"
    return canonical_json(selector)


def _components_for_series(scoring_cfg: dict[str, Any], series_label: str | None) -> list[dict[str, Any]]:
    if not isinstance(scoring_cfg, dict):
        return []
    default_block = scoring_cfg.get("default") if isinstance(scoring_cfg.get("default"), dict) else {}
    series_map = scoring_cfg.get("series") if isinstance(scoring_cfg.get("series"), dict) else {}
    series_block = series_map.get(series_label) if isinstance(series_map, dict) else None
    composite = series_block.get("composite") if isinstance(series_block, dict) else None
    if not isinstance(composite, dict):
        composite = default_block.get("composite") if isinstance(default_block, dict) else None
    components = composite.get("components") if isinstance(composite, dict) else None
    if not isinstance(components, list):
        return []
    return _flatten_components([c for c in components if isinstance(c, dict)])


def _baseline_provider_device_for_series(
    scoring_cfg: dict[str, Any],
    series_label: str | None,
) -> tuple[str | None, str | None]:
    if not isinstance(scoring_cfg, dict):
        return None, None
    default_block = scoring_cfg.get("default") if isinstance(scoring_cfg.get("default"), dict) else {}
    series_map = scoring_cfg.get("series") if isinstance(scoring_cfg.get("series"), dict) else {}
    series_block = series_map.get(series_label) if isinstance(series_map, dict) else None
    baseline = series_block.get("baseline") if isinstance(series_block, dict) else None
    if not isinstance(baseline, dict):
        baseline = default_block.get("baseline") if isinstance(default_block, dict) else None
    if not isinstance(baseline, dict):
        return None, None
    provider = baseline.get("provider") if isinstance(baseline.get("provider"), str) else None
    device = baseline.get("device") if isinstance(baseline.get("device"), str) else None
    return provider, device


def _is_baseline_row_for_series(
    scoring_cfg: dict[str, Any],
    series_label: str | None,
    row: dict[str, Any],
) -> bool:
    base_provider, base_device = _baseline_provider_device_for_series(scoring_cfg, series_label)
    if not base_provider or not base_device:
        return False
    return row.get("provider") == base_provider and row.get("device") == base_device


def _matching_components_for_row(
    scoring_cfg: dict[str, Any],
    series_label: str | None,
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    bench = derive_benchmark_name(row)
    out: list[dict[str, Any]] = []
    for comp in _components_for_series(scoring_cfg, series_label):
        if not _component_matches_benchmark(comp, bench):
            continue
        selector = comp.get("selector") if isinstance(comp.get("selector"), dict) else None
        if not _row_param_matches(selector, row):
            continue
        metric = comp.get("metric")
        if not isinstance(metric, str):
            continue
        out.append(comp)
    return out


def load_baselines_config(root: str) -> dict[str, Any]:
    """Load baseline configuration from scripts/scoring.json.

    Expected shape (top-level):
      {
        "series": {
          "v0.4": { "provider": "ibm", "device": "ibm_torino" }
        },
        "default": { "provider": "ibm", "device": "ibm_torino" },
        "composites": { ... }  # optional, ignored here
      }
    """
    scoring_path = os.path.join(root, "scripts", "scoring.json")
    try:
        with open(scoring_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        print("Error: scripts/scoring.json not found", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Warning: failed to load scoring config: {e}", file=sys.stderr)
        return {}


def compute_baseline_averages_by_series(
    flat_rows: list[dict[str, Any]],
    row_series: dict[int, str],
    baselines_cfg: dict[str, Any],
) -> tuple[dict[str, dict[tuple[str, str, str], float]], list[str]]:
    series_list = sorted(set(row_series.values()))
    baseline_avg_by_series: dict[str, dict[tuple[str, str, str], float]] = {}
    summary: list[str] = []

    # Build major->latest-series map from observed series labels.
    major_to_latest_series: dict[int, str] = {}
    major_to_latest_ver: dict[int, tuple[int, ...]] = {}
    for series in series_list:
        ver = _parse_series_label(series)
        if not ver:
            continue
        major = ver[0]
        cur_best = major_to_latest_ver.get(major)
        if cur_best is None or ver > cur_best:
            major_to_latest_ver[major] = ver
            major_to_latest_series[major] = series

    # Compute one baseline map per major:
    # latest baseline value per (benchmark, metric, selector), then reuse for every minor.
    baseline_by_major: dict[int, dict[tuple[str, str, str], float]] = {}
    baseline_choice_by_major: dict[int, tuple[str | None, str | None, str | None]] = {}
    for major, ref_series in sorted(major_to_latest_series.items()):
        series_block = (baselines_cfg.get("series", {}) or {}).get(ref_series, {}) if isinstance(baselines_cfg, dict) else {}
        baseline_obj = series_block.get("baseline") or (baselines_cfg.get("default", {}) or {}).get("baseline", {})
        base_provider = (baseline_obj or {}).get("provider")
        base_device = (baseline_obj or {}).get("device")
        baseline_choice_by_major[major] = (base_provider, base_device, ref_series)

        if not base_provider or not base_device:
            baseline_by_major[major] = {}
            continue

        selected = [
            r for r in flat_rows
            if r.get("provider") == base_provider
            and r.get("device") == base_device
            and _series_major(row_series.get(id(r))) == major
        ]

        # Keep the latest timestamped baseline row per (benchmark, metric, selector).
        # Collect component definitions from all observed minors in this major so
        # baseline coverage is major-wide even if minor configs differ.
        latest_baseline: dict[tuple[str, str, str], tuple[datetime | None, float]] = {}
        major_series_labels = [s for s in series_list if _series_major(s) == major]
        components: list[dict[str, Any]] = []
        for s in major_series_labels:
            components.extend(_components_for_series(baselines_cfg, s))
        if not components:
            components = _components_for_series(baselines_cfg, ref_series)
        for comp in components:
            metric = comp.get("metric")
            if not isinstance(metric, str):
                continue
            selector = comp.get("selector") if isinstance(comp.get("selector"), dict) else None
            selector_fp = _selector_fingerprint(selector)

            for r in selected:
                bench = derive_benchmark_name(r)
                if not _component_matches_benchmark(comp, bench):
                    continue
                if not _row_param_matches(selector, r):
                    continue
                results = r.get("results") if isinstance(r.get("results"), dict) else {}
                candidate_metrics = [metric]
                for part in _derived_from_components(comp):
                    part_metric = part.get("metric")
                    if isinstance(part_metric, str) and part_metric:
                        candidate_metrics.append(part_metric)

                ts = parse_timestamp(r.get("timestamp", ""))
                for m in candidate_metrics:
                    if m not in results:
                        continue
                    try:
                        num = float(results.get(m))
                    except Exception:
                        continue
                    if not (num == num):  # NaN
                        continue
                    key = (bench, m, selector_fp)
                    prev = latest_baseline.get(key)
                    if prev is None:
                        latest_baseline[key] = (ts, num)
                        continue
                    prev_ts, _prev_num = prev
                    if prev_ts is None and ts is not None:
                        latest_baseline[key] = (ts, num)
                    elif prev_ts is not None and ts is not None and ts > prev_ts:
                        latest_baseline[key] = (ts, num)

        baseline_by_major[major] = {key: val for key, (_ts, val) in latest_baseline.items()}

    for series in series_list:
        major = _series_major(series)
        if major is not None and major in baseline_by_major:
            baseline_avg_by_series[series] = baseline_by_major[major]
            base_provider, base_device, ref_series = baseline_choice_by_major.get(major, (None, None, None))
            if base_provider and base_device:
                summary.append(
                    f"{series}: {base_provider}/{base_device} (major {major}, ref {ref_series}, latest-per-key)"
                )
            else:
                summary.append(f"{series}: (no baseline configured for major {major})")
            continue

        # Unknown/non-version series fallback to legacy per-series lookup.
        series_block = (baselines_cfg.get("series", {}) or {}).get(series, {}) if isinstance(baselines_cfg, dict) else {}
        baseline_obj = series_block.get("baseline") or (baselines_cfg.get("default", {}) or {}).get("baseline", {})
        base_provider = (baseline_obj or {}).get("provider")
        base_device = (baseline_obj or {}).get("device")
        if not base_provider or not base_device:
            baseline_avg_by_series[series] = {}
            summary.append(f"{series}: (no baseline configured)")
            continue
        selected = [
            r for r in flat_rows
            if r.get("provider") == base_provider and r.get("device") == base_device and row_series.get(id(r)) == series
        ]
        latest_baseline: dict[tuple[str, str, str], tuple[datetime | None, float]] = {}
        components = _components_for_series(baselines_cfg, series)
        for comp in components:
            metric = comp.get("metric")
            if not isinstance(metric, str):
                continue
            selector = comp.get("selector") if isinstance(comp.get("selector"), dict) else None
            selector_fp = _selector_fingerprint(selector)
            for r in selected:
                bench = derive_benchmark_name(r)
                if not _component_matches_benchmark(comp, bench):
                    continue
                if not _row_param_matches(selector, r):
                    continue
                results = r.get("results") if isinstance(r.get("results"), dict) else {}
                candidate_metrics = [metric]
                for part in _derived_from_components(comp):
                    part_metric = part.get("metric")
                    if isinstance(part_metric, str) and part_metric:
                        candidate_metrics.append(part_metric)
                ts = parse_timestamp(r.get("timestamp", ""))
                for m in candidate_metrics:
                    if m not in results:
                        continue
                    num = _coerce_float(results.get(m))
                    if num is None:
                        continue
                    key = (bench, m, selector_fp)
                    prev = latest_baseline.get(key)
                    if prev is None:
                        latest_baseline[key] = (ts, num)
                        continue
                    prev_ts, _prev_num = prev
                    if prev_ts is None and ts is not None:
                        latest_baseline[key] = (ts, num)
                    elif prev_ts is not None and ts is not None and ts > prev_ts:
                        latest_baseline[key] = (ts, num)
        baseline_avg_by_series[series] = {key: val for key, (_ts, val) in latest_baseline.items()}
        summary.append(f"{series}: {base_provider}/{base_device} (latest-per-key)")

    return baseline_avg_by_series, summary


def compute_and_attach_metriq_scores(
    flat_rows: list[dict[str, Any]],
    row_series: dict[int, str],
    baseline_avg_by_series: dict[str, dict[tuple[str, str, str], float]],
    scoring_cfg: dict[str, Any],
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

        matching_components = _matching_components_for_row(scoring_cfg, series, r)
        if not matching_components:
            continue
        # For a given row, only compute normalized scores for metrics explicitly configured
        # by matching components (benchmark + selector).
        metric_to_comp: dict[str, dict[str, Any]] = {}
        for comp in matching_components:
            metric = comp.get("metric")
            if not isinstance(metric, str) or not metric:
                continue
            metric_to_comp[metric] = comp
        target_metrics = list(metric_to_comp.keys())

        for metric in target_metrics:
            comp = metric_to_comp.get(metric, {})
            selector = comp.get("selector") if isinstance(comp.get("selector"), dict) else None
            selector_fp = _selector_fingerprint(selector)

            if metric in results:
                val = results.get(metric)
                try:
                    v = float(val)
                except Exception:
                    continue
                if not (v == v):  # NaN
                    continue
                base = baseline_avg.get((bench, metric, selector_fp))
                if base is None:
                    base = _fallback_baseline_average(
                        series, bench, metric, selector_fp, baseline_avg_by_series
                    )
                if base is None:
                    continue
                direction = str(dir_map.get(metric, "higher")).lower()
                score: float | None = None
                try:
                    if direction == "lower":
                        if v > 0:
                            score = (base / v) * 100.0
                    else:
                        if base > 0:
                            score = (v / base) * 100.0
                except Exception:
                    score = None
            else:
                score = _compute_derived_normalized_score(
                    r,
                    series,
                    selector_fp,
                    comp,
                    baseline_avg_by_series,
                )
            if score is not None and (score == score) and score not in (float("inf"), float("-inf")):
                scores[metric] = score
        if not scores:
            continue

        # Attach per-metric normalized scores for downstream composite aggregation.
        # This preserves all computed normalized metrics even for multi-metric benchmarks.
        r["normalized_scores"] = scores

        # Choose a scalar metriq_score when possible:
        #  - If exactly one configured metric applies to this row, expose it as metriq_score
        if len(scores) == 1:
            r["metriq_score"] = next(iter(scores.values()))


def load_scoring_config(root: str) -> dict[str, Any]:
    """Load scoring configuration (baselines + composite) from scripts/scoring.json.

    Expected shape:
      {
        "series": {
          "vX.Y": {
            "baseline": { "provider": str, "device": str },
            "composite": { "components": [ ... ] }
          },
          ...
        },
        "default": {
          "baseline": { "provider": str, "device": str },
          "composite": { "components": [ ... ] }
        }
      }
    """
    scoring_path = os.path.join(root, "scripts", "scoring.json")
    try:
        with open(scoring_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        print("Error: scripts/scoring.json not found", file=sys.stderr)
    except Exception as e:
        print(f"Warning: failed to load scoring.json: {e}", file=sys.stderr)
    return {}


def _validate_components_list(components: list[dict[str, Any]], ctx: str) -> None:
    total = 0.0
    for i, comp in enumerate(components):
        if not isinstance(comp, dict):
            raise ValueError(f"Invalid component at index {i} in {ctx}: expected object")
        children = comp.get("components")
        w_raw = comp.get("weight")
        try:
            w = _parse_weight(w_raw)
        except Exception:
            raise ValueError(f"Invalid weight for component {i} in {ctx}: {w_raw}")
        if w < 0:
            raise ValueError(f"Negative weight for component {i} in {ctx}: {w_raw}")

        if isinstance(children, list):
            _validate_components_list(children, ctx=f"{ctx}.components[{i}]")
        total += w

    if components:
        # Weight lists are expected to represent convex combinations.
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Component weights must sum to 1.0 in {ctx}: got {total}")


def validate_scoring_config(scoring_cfg: dict[str, Any]) -> None:
    """Ensure each series' composite components weights sum to 1.0.

    Validates both default.composite.components and series.<v>.composite.components
    (when present). Raises ValueError on any violation.
    """
    if not isinstance(scoring_cfg, dict):
        raise ValueError("Invalid scoring config: expected object")

    default = scoring_cfg.get("default")
    if isinstance(default, dict):
        comp = default.get("composite")
        if isinstance(comp, dict) and isinstance(comp.get("components"), list):
            _validate_components_list(comp["components"], ctx="default.composite")

    series_map = scoring_cfg.get("series")
    if isinstance(series_map, dict):
        for label, block in series_map.items():
            if not isinstance(block, dict):
                continue
            comp = block.get("composite")
            if isinstance(comp, dict) and isinstance(comp.get("components"), list):
                _validate_components_list(comp["components"], ctx=f"series.{label}.composite")


def _row_param_matches(selector: dict[str, Any] | None, row: dict[str, Any]) -> bool:
    if not selector:
        return True
    params = row.get("params") if isinstance(row.get("params"), dict) else {}
    for k, v in selector.items():
        if params.get(k) != v:
            return False
    return True

def _get_normalized_metric_value(
    row: dict[str, Any],
    metric: str,
    series_label: str | None,
    selector_fp: str,
    baseline_avg_by_series: dict[str, dict[tuple[str, str, str], float]],
) -> float | None:
    # Prefer precomputed normalized scores
    norm = row.get("normalized_scores")
    if isinstance(norm, dict) and metric in norm and norm[metric] is not None:
        try:
            return float(norm[metric])
        except Exception:
            pass

    # Fallback: compute ad-hoc from row's raw result using baseline averages
    results = row.get("results") if isinstance(row.get("results"), dict) else {}
    if metric not in results:
        return None
    try:
        v = float(results[metric])
    except Exception:
        return None
    if not (v == v):  # NaN
        return None
    bench = derive_benchmark_name(row)
    baseline_avg = baseline_avg_by_series.get(series_label or "", {})
    base = baseline_avg.get((bench, metric, selector_fp))
    if base is None:
        base = _fallback_baseline_average(series_label, bench, metric, selector_fp, baseline_avg_by_series)
    if base is None:
        return None
    dir_map = row.get("directions") if isinstance(row.get("directions"), dict) else {}
    direction = str(dir_map.get(metric, "higher")).lower()
    try:
        if direction == "lower":
            if v > 0:
                return (base / v) * 100.0
        else:
            if base > 0:
                return (v / base) * 100.0
    except Exception:
        return None
    return None


def _get_raw_metric_value(row: dict[str, Any], metric: str) -> float | None:
    """Return the raw metric value from row.results when present, else None."""
    results = row.get("results") if isinstance(row.get("results"), dict) else {}
    if metric not in results:
        return None
    return _coerce_float(results.get(metric))


def _pick_latest_metric_row(
    candidates: list[dict[str, Any]],
    value_key: str,
) -> tuple[dict[str, Any] | None, datetime | None]:
    """Pick the latest timestamped row containing value_key."""
    picked = None
    picked_ts = None
    for cand in candidates:
        if value_key not in cand:
            continue
        ts = parse_timestamp(cand.get("timestamp", ""))
        if ts is None:
            continue
        if picked is None or picked_ts is None or ts > picked_ts:
            picked = cand
            picked_ts = ts
    return picked, picked_ts


def compute_device_composite_scores(
    flat_rows: list[dict[str, Any]],
    row_series: dict[int, str],
    baseline_avg_by_series: dict[str, dict[tuple[str, str, str], float]],
    scoring_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute per-(provider, device) composite Metriq Score using series-scoped configs.

    For each device, choose its latest series (by most recent timestamp among its rows),
    then compute the composite using the series-specific components (fallback to default).
    When selecting rows for each component, include all rows in the same semantic major
    version as the picked series (e.g., if picked series is v0.6.1, include v0.4/v0.5/v0.6.1).

    Returns a list of records:
      { provider, device, metriq_score, components: { ... }, series }
    where each component includes explicit availability fields for both
    normalized and raw values.
    """

    # group rows by (provider, device)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in flat_rows:
        provider = r.get("provider")
        device = r.get("device")
        if not provider or not device:
            continue
        grouped.setdefault((provider, device), []).append(r)

    out: list[dict[str, Any]] = []
    series_cfg_map = scoring_cfg.get("series", {}) if isinstance(scoring_cfg, dict) else {}
    default_composite = ((scoring_cfg.get("default", {}) or {}).get("composite", {})
                         if isinstance(scoring_cfg, dict) else {})
    for (provider, device), rows in grouped.items():
        # pick device's latest series by timestamp
        latest_ts = None
        picked_series = None
        for r in rows:
            ts = parse_timestamp(r.get("timestamp", ""))
            if ts is None:
                continue
            s = row_series.get(id(r))
            if picked_series is None or latest_ts is None or ts > latest_ts:
                latest_ts = ts
                picked_series = s

        picked_major = _series_major(picked_series)
        series_block = (series_cfg_map.get(picked_series, {}) if isinstance(series_cfg_map, dict) else {})
        composite_cfg = series_block.get("composite") or default_composite
        components_cfg = composite_cfg.get("components") if isinstance(composite_cfg, dict) else None
        if not isinstance(components_cfg, list) or not components_cfg:
            # No components configured; skip this device
            continue
        components_flat = _flatten_components([c for c in components_cfg if isinstance(c, dict)])
        breakdown: dict[str, dict[str, Any]] = {}
        numerator = 0.0
        sum_w_defined = 0.0

        for comp in components_flat:
            if not isinstance(comp, dict):
                continue
            bench_field = comp.get("benchmark")
            # Allow a single name or a list of names, plus optional aliases
            allowed_names: set[str] = set()
            if isinstance(bench_field, str):
                allowed_names.add(bench_field)
            elif isinstance(bench_field, list):
                for b in bench_field:
                    if isinstance(b, str):
                        allowed_names.add(b)
            aliases = comp.get("aliases")
            if isinstance(aliases, list):
                for a in aliases:
                    if isinstance(a, str):
                        allowed_names.add(a)
            metric = comp.get("metric")
            group_label = comp.get("_group_label")
            group_weight = float(comp.get("_group_weight", 1.0))
            sub_weight = float(comp.get("_sub_weight", 0.0))
            weight = float(comp.get("_effective_weight", _parse_weight(comp.get("weight", 0.0))))
            selector = comp.get("selector") if isinstance(comp.get("selector"), dict) else None
            # Prefer the primary benchmark name for label when available
            primary_bench: str | None = None
            if isinstance(bench_field, str):
                primary_bench = bench_field
            elif isinstance(bench_field, list) and bench_field:
                first = bench_field[0]
                if isinstance(first, str):
                    primary_bench = first

            label = comp.get("label")
            if not label:
                label = f"{primary_bench}:{metric}" if primary_bench and metric else "component"
            # Always include every component's weight in the denominator; if a
            # component is missing for this device, its normalized contribution is 0.
            sum_w_defined += weight

            # Filter rows by benchmark, selector, and major-version group.
            # Keep candidates if either normalized or raw value is available.
            matches: list[dict[str, Any]] = []
            for r in rows:
                # Match benchmark by any allowed name (if provided)
                if allowed_names and derive_benchmark_name(r) not in allowed_names:
                    continue
                if not _row_param_matches(selector, r):
                    continue
                series_label = row_series.get(id(r))
                if picked_major is not None:
                    if _series_major(series_label) != picked_major:
                        continue
                elif series_label != picked_series:
                    continue
                selector_fp = _selector_fingerprint(selector)
                normalized_val = _get_normalized_metric_value(
                    r, metric, series_label, selector_fp, baseline_avg_by_series
                )
                if normalized_val is not None and _is_baseline_row_for_series(scoring_cfg, series_label, r):
                    # Keep baseline components anchored at 100 in platform composites.
                    normalized_val = 100.0
                raw_val = _get_raw_metric_value(r, metric)
                if normalized_val is None and raw_val is None:
                    continue
                r_copy = dict(r)
                if normalized_val is not None:
                    r_copy["_normalized_val"] = float(normalized_val)
                if raw_val is not None:
                    r_copy["_raw_val"] = float(raw_val)
                matches.append(r_copy)

            picked_norm, _ = _pick_latest_metric_row(matches, "_normalized_val")
            picked_raw, _ = _pick_latest_metric_row(matches, "_raw_val")

            normalized_value = (
                float(picked_norm.get("_normalized_val"))
                if picked_norm is not None and picked_norm.get("_normalized_val") is not None
                else None
            )
            raw_value = (
                float(picked_raw.get("_raw_val"))
                if picked_raw is not None and picked_raw.get("_raw_val") is not None
                else None
            )
            normalized_ts = picked_norm.get("timestamp") if picked_norm is not None else None
            raw_ts = picked_raw.get("timestamp") if picked_raw is not None else None

            if normalized_value is not None:
                numerator += weight * normalized_value

            breakdown[label] = {
                "metric": metric,
                "weight": weight,
                "group": group_label,
                "group_weight": group_weight,
                "sub_weight": sub_weight,
                # Backward-compatible key for normalized timestamp.
                "timestamp": normalized_ts,
                # Explicit availability fields for UI rendering.
                "normalized": normalized_value,
                "normalized_available": normalized_value is not None,
                "normalized_timestamp": normalized_ts,
                "raw": raw_value,
                "raw_available": raw_value is not None,
                "raw_timestamp": raw_ts,
            }

        # Denominator is the sum of all defined weights; missing components
        # contribute 0 to the numerator but still count in the denominator.
        if sum_w_defined > 0.0:
            metriq_score = float(numerator / sum_w_defined)
        else:
            metriq_score = None

        out.append(
            {
                "provider": provider,
                "device": device,
                "metriq_score": metriq_score,
                "components": breakdown,
                "series": picked_series,
            }
        )

    return out
