from datetime import datetime
import json
import os
from typing import Any, NamedTuple

from etl import (
    canonical_device_name,
    canonical_json,
    canonical_provider_name,
    derive_benchmark_name,
    parse_timestamp,
)


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


def _weighted_harmonic_mean(pairs: list[tuple[float, float]]) -> float | None:
    """Weighted harmonic mean of (weight, value) pairs.

    Returns 0.0 if any weighted value is <= 0 (the harmonic mean's limit as a
    component approaches zero), and None if no positively weighted values exist.
    """
    total_w = 0.0
    inv_sum = 0.0
    for w, v in pairs:
        if w <= 0:
            continue
        if v <= 0:
            return 0.0
        total_w += w
        inv_sum += w / v
    if total_w <= 0 or inv_sum <= 0:
        return None
    return total_w / inv_sum


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
      - _group_label, _group_weight, _sub_weight, _effective_weight, _group_aggregation
    """
    flat: list[dict[str, Any]] = []
    for comp in components:
        if not isinstance(comp, dict):
            continue
        group_children = comp.get("components")
        if isinstance(group_children, list):
            group_label = comp.get("label") or comp.get("benchmark") or "group"
            group_weight = _parse_weight(comp.get("weight", 0.0))
            group_aggregation = str(comp.get("aggregation", "arithmetic")).lower()
            for child in group_children:
                if not isinstance(child, dict):
                    continue
                sub_weight = _parse_weight(child.get("weight", 0.0))
                merged = dict(child)
                merged["_group_label"] = group_label
                merged["_group_weight"] = group_weight
                merged["_sub_weight"] = sub_weight
                merged["_effective_weight"] = group_weight * sub_weight
                merged["_group_aggregation"] = group_aggregation
                flat.append(merged)
            continue

        sub_weight = _parse_weight(comp.get("weight", 0.0))
        merged = dict(comp)
        merged["_group_label"] = None
        merged["_group_weight"] = 1.0
        merged["_sub_weight"] = sub_weight
        merged["_effective_weight"] = sub_weight
        merged["_group_aggregation"] = "arithmetic"
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


def _component_benchmarks(comp: dict[str, Any]) -> set[str]:
    benchmark = comp.get("benchmark")
    names = {benchmark} if isinstance(benchmark, str) else set()
    if isinstance(benchmark, list):
        names.update(name for name in benchmark if isinstance(name, str))
    aliases = comp.get("aliases")
    if isinstance(aliases, list):
        names.update(alias for alias in aliases if isinstance(alias, str))
    return names


def _primary_benchmark(comp: dict[str, Any]) -> str | None:
    benchmark = comp.get("benchmark")
    if isinstance(benchmark, str):
        return benchmark
    if isinstance(benchmark, list):
        return next((name for name in benchmark if isinstance(name, str)), None)
    return None


def _component_matches_benchmark(comp: dict[str, Any], bench: str) -> bool:
    return bench in _component_benchmarks(comp)


def _selector_fingerprint(selector: dict[str, Any] | None) -> str:
    if not selector:
        return "null"
    return canonical_json(selector)


def _series_setting(
    scoring_cfg: dict[str, Any], series_label: str | None, name: str
) -> dict[str, Any]:
    """Return a series setting, falling back to the default block."""
    series_map = scoring_cfg.get("series")
    series_block = series_map.get(series_label) if isinstance(series_map, dict) else None
    value = series_block.get(name) if isinstance(series_block, dict) else None
    if isinstance(value, dict):
        return value
    default = scoring_cfg.get("default")
    value = default.get(name) if isinstance(default, dict) else None
    return value if isinstance(value, dict) else {}


def _components_for_series(
    scoring_cfg: dict[str, Any], series_label: str | None
) -> list[dict[str, Any]]:
    components = _series_setting(scoring_cfg, series_label, "composite").get("components")
    if not isinstance(components, list):
        return []
    return _flatten_components([c for c in components if isinstance(c, dict)])


def _baseline_provider_device_for_series(
    scoring_cfg: dict[str, Any],
    series_label: str | None,
) -> tuple[str | None, str | None]:
    baseline = _series_setting(scoring_cfg, series_label, "baseline")
    provider = baseline.get("provider") if isinstance(baseline.get("provider"), str) else None
    device = baseline.get("device") if isinstance(baseline.get("device"), str) else None
    if provider is None or device is None:
        return provider, device
    provider = canonical_provider_name(provider)
    return provider, canonical_device_name(provider, device)


def baseline_metadata_for_latest_series(
    scoring_cfg: dict[str, Any],
    series_labels: list[str],
) -> dict[str, str] | None:
    """Return the canonical baseline configured for the latest observed series."""
    latest_series: str | None = None
    latest_version: tuple[int, ...] | None = None
    for series in series_labels:
        version = _parse_series_label(series)
        if version is None:
            continue
        if latest_version is None or version > latest_version:
            latest_series = series
            latest_version = version

    if latest_series is None:
        return None
    provider, device = _baseline_provider_device_for_series(scoring_cfg, latest_series)
    if provider is None or device is None:
        return None
    return {
        "provider": provider,
        "device": device,
        "series": latest_series,
    }


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
    components: list[dict[str, Any]],
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    bench = derive_benchmark_name(row)
    out: list[dict[str, Any]] = []
    for comp in components:
        if not _component_matches_benchmark(comp, bench):
            continue
        if not _component_param_matches(comp, row):
            continue
        metric = comp.get("metric")
        if not isinstance(metric, str) or not metric:
            continue
        out.append(comp)
    return out


def _component_candidate_metrics(comp: dict[str, Any]) -> list[str]:
    metric = comp.get("metric")
    if not isinstance(metric, str) or not metric:
        return []
    out: list[str] = [metric]
    seen: set[str] = {metric}
    for part in _derived_from_components(comp):
        part_metric = part.get("metric")
        if not isinstance(part_metric, str) or not part_metric:
            continue
        if part_metric in seen:
            continue
        seen.add(part_metric)
        out.append(part_metric)
    return out


def _latest_baseline_values_for_components(
    selected_rows: list[dict[str, Any]],
    components: list[dict[str, Any]],
) -> dict[tuple[str, str, str], float]:
    latest_baseline: dict[tuple[str, str, str], tuple[datetime | None, float]] = {}
    for comp in components:
        metrics = _component_candidate_metrics(comp)
        if not metrics:
            continue
        selector = comp.get("selector") if isinstance(comp.get("selector"), dict) else None
        selector_fp = _selector_fingerprint(selector)

        for r in selected_rows:
            bench = derive_benchmark_name(r)
            if not _component_matches_benchmark(comp, bench):
                continue
            if not _component_param_matches(comp, r):
                continue
            results = r.get("results") if isinstance(r.get("results"), dict) else {}
            ts = parse_timestamp(r.get("timestamp", ""))
            for metric in metrics:
                if metric not in results:
                    continue
                num = _coerce_float(results.get(metric))
                if num is None:
                    continue
                key = (bench, metric, selector_fp)
                prev = latest_baseline.get(key)
                if prev is None:
                    latest_baseline[key] = (ts, num)
                    continue
                prev_ts, _prev_num = prev
                if prev_ts is None and ts is not None:
                    latest_baseline[key] = (ts, num)
                elif prev_ts is not None and ts is not None and ts > prev_ts:
                    latest_baseline[key] = (ts, num)
    return {key: val for key, (_ts, val) in latest_baseline.items()}


def _baseline_summary_line(
    series: str,
    base_provider: str | None,
    base_device: str | None,
    *,
    major: int | None = None,
    ref_series: str | None = None,
) -> str:
    if not base_provider or not base_device:
        if major is not None:
            return f"{series}: (no baseline configured for major {major})"
        return f"{series}: (no baseline configured)"
    if major is not None and ref_series is not None:
        return f"{series}: {base_provider}/{base_device} (major {major}, ref {ref_series}, latest-per-key)"
    return f"{series}: {base_provider}/{base_device} (latest-per-key)"


def compute_baseline_averages_by_series(
    flat_rows: list[dict[str, Any]],
    row_series: dict[int, str],
    baselines_cfg: dict[str, Any],
) -> tuple[dict[str, dict[tuple[str, str, str], float]], list[str]]:
    series_list = sorted(set(row_series.values()))
    baseline_avg_by_series: dict[str, dict[tuple[str, str, str], float]] = {}
    summary: list[str] = []
    row_major_by_id = {rid: _series_major(series_label) for rid, series_label in row_series.items()}

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
        base_provider, base_device = _baseline_provider_device_for_series(baselines_cfg, ref_series)
        baseline_choice_by_major[major] = (base_provider, base_device, ref_series)

        if not base_provider or not base_device:
            baseline_by_major[major] = {}
            continue

        selected = [
            r for r in flat_rows
            if r.get("provider") == base_provider
            and r.get("device") == base_device
            and row_major_by_id.get(id(r)) == major
        ]

        # Keep the latest timestamped baseline row per (benchmark, metric, selector).
        # Collect component definitions from all observed minors in this major so
        # baseline coverage is major-wide even if minor configs differ.
        major_series_labels = [s for s in series_list if _series_major(s) == major]
        components: list[dict[str, Any]] = []
        for s in major_series_labels:
            components.extend(_components_for_series(baselines_cfg, s))
        if not components:
            components = _components_for_series(baselines_cfg, ref_series)
        baseline_by_major[major] = _latest_baseline_values_for_components(selected, components)

    # Carry missing baseline keys forward from earlier score majors. This is the
    # same fallback policy normalization used to apply one lookup at a time.
    inherited: dict[tuple[str, str, str], float] = {}
    for major in sorted(baseline_by_major):
        inherited = {**inherited, **baseline_by_major[major]}
        baseline_by_major[major] = inherited

    for series in series_list:
        major = _series_major(series)
        if major is not None and major in baseline_by_major:
            baseline_avg_by_series[series] = baseline_by_major[major]
            base_provider, base_device, ref_series = baseline_choice_by_major.get(major, (None, None, None))
            summary.append(
                _baseline_summary_line(
                    series,
                    base_provider,
                    base_device,
                    major=major,
                    ref_series=ref_series,
                )
            )
            continue

        # Unknown/non-version series fallback to legacy per-series lookup.
        base_provider, base_device = _baseline_provider_device_for_series(baselines_cfg, series)
        if not base_provider or not base_device:
            baseline_avg_by_series[series] = {}
            summary.append(_baseline_summary_line(series, base_provider, base_device))
            continue
        selected = [
            r for r in flat_rows
            if r.get("provider") == base_provider and r.get("device") == base_device and row_series.get(id(r)) == series
        ]
        components = _components_for_series(baselines_cfg, series)
        baseline_avg_by_series[series] = _latest_baseline_values_for_components(selected, components)
        summary.append(_baseline_summary_line(series, base_provider, base_device))

    return baseline_avg_by_series, summary


def compute_and_attach_metriq_scores(
    flat_rows: list[dict[str, Any]],
    row_series: dict[int, str],
    baseline_avg_by_series: dict[str, dict[tuple[str, str, str], float]],
    scoring_cfg: dict[str, Any],
) -> None:
    component_cache: dict[str | None, list[dict[str, Any]]] = {}
    for r in flat_rows:
        results = r.get("results") if isinstance(r.get("results"), dict) else {}
        if not results:
            continue
        series = row_series.get(id(r))
        scores: dict[str, float] = {}

        if series not in component_cache:
            component_cache[series] = _components_for_series(scoring_cfg, series)
        matching_components = _matching_components_for_row(component_cache[series], r)
        if not matching_components:
            continue
        # For a given row, only compute normalized scores for metrics explicitly configured
        # by matching components (benchmark + selector).
        metric_to_comp = {comp["metric"]: comp for comp in matching_components}

        for metric, comp in metric_to_comp.items():
            selector = comp.get("selector") if isinstance(comp.get("selector"), dict) else None
            selector_fp = _selector_fingerprint(selector)

            if metric in results:
                score = _normalized_ratio(
                    results[metric],
                    _get_baseline_metric_value(
                        r, metric, series, selector_fp, baseline_avg_by_series
                    ),
                    _metric_direction(r, metric),
                )
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

        # Expose a scalar metriq_score:
        #  - single-metric rows map directly
        #  - multi-metric rows use a weighted mean by component effective weights,
        #    arithmetic by default or harmonic when the components' group is
        #    configured with "aggregation": "harmonic" (e.g. EPLG)
        if len(scores) == 1:
            r["metriq_score"] = next(iter(scores.values()))
        else:
            pairs: list[tuple[float, float]] = []
            harmonic = False
            for metric, metric_score in scores.items():
                comp = metric_to_comp.get(metric, {})
                if comp.get("_group_aggregation") == "harmonic":
                    harmonic = True
                weight = float(comp.get("_effective_weight", 0.0))
                if weight <= 0.0:
                    continue
                pairs.append((weight, metric_score))
            if harmonic:
                hm = _weighted_harmonic_mean(pairs)
                if hm is not None:
                    r["metriq_score"] = hm
            else:
                weighted_sum = sum(w * s for w, s in pairs)
                weight_total = sum(w for w, _s in pairs)
                if weight_total > 0.0:
                    r["metriq_score"] = weighted_sum / weight_total


def load_scoring_config(root: str) -> dict[str, Any]:
    """Load the required scoring configuration."""
    scoring_path = os.path.join(root, "scripts", "scoring.json")
    with open(scoring_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid scoring config: expected object")
    return data


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

        agg = comp.get("aggregation")
        if agg is not None:
            if not isinstance(agg, str) or agg.lower() not in ("arithmetic", "harmonic"):
                raise ValueError(
                    f"Invalid aggregation for component {i} in {ctx}: {agg!r} "
                    "(expected 'arithmetic' or 'harmonic')"
                )
            if not isinstance(children, list):
                raise ValueError(
                    f"'aggregation' is only valid on group components (index {i} in {ctx})"
                )

        selector_alternatives = comp.get("selector_alternatives")
        if selector_alternatives is not None:
            if not isinstance(comp.get("selector"), dict) or not comp["selector"]:
                raise ValueError(
                    f"Selector alternatives require a non-empty primary selector "
                    f"(index {i} in {ctx})"
                )
            if not isinstance(selector_alternatives, list) or not selector_alternatives:
                raise ValueError(
                    f"Invalid selector alternatives for component {i} in {ctx}: "
                    "expected a non-empty list"
                )
            if any(not isinstance(item, dict) or not item for item in selector_alternatives):
                raise ValueError(
                    f"Invalid selector alternative for component {i} in {ctx}: "
                    "expected non-empty objects"
                )

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


def _component_param_matches(comp: dict[str, Any], row: dict[str, Any]) -> bool:
    """Match a row against a component's primary or alternative selectors."""
    selector = comp.get("selector") if isinstance(comp.get("selector"), dict) else None
    if _row_param_matches(selector, row):
        return True
    alternatives = comp.get("selector_alternatives")
    if not isinstance(alternatives, list):
        return False
    return any(
        _row_param_matches(alternative, row)
        for alternative in alternatives
        if isinstance(alternative, dict) and alternative
    )


def _normalized_ratio(value: Any, baseline: Any, direction: str) -> float | None:
    """Normalize one raw value against its baseline."""
    value_num = _coerce_float(value)
    baseline_num = _coerce_float(baseline)
    if value_num is None or baseline_num is None:
        return None
    if value_num in (float("inf"), float("-inf")) or baseline_num in (
        float("inf"),
        float("-inf"),
    ):
        return None
    if direction == "lower":
        return (baseline_num / value_num) * 100.0 if value_num > 0 else None
    return (value_num / baseline_num) * 100.0 if baseline_num > 0 else None


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
        value = _coerce_float(norm[metric])
        if value is not None:
            return value

    return _normalized_ratio(
        _get_raw_metric_value(row, metric),
        _get_baseline_metric_value(
            row, metric, series_label, selector_fp, baseline_avg_by_series
        ),
        _metric_direction(row, metric),
    )


def _get_baseline_metric_value(
    row: dict[str, Any],
    metric: str,
    series_label: str | None,
    selector_fp: str,
    baseline_avg_by_series: dict[str, dict[tuple[str, str, str], float]],
) -> float | None:
    """Return the baseline device's raw value for the same (benchmark, metric, selector).

    This is the denominator that _get_normalized_metric_value would use. It is
    exposed separately so composite scoring can aggregate raw values across
    circuit widths before normalizing.
    """
    bench = derive_benchmark_name(row)
    baseline_avg = baseline_avg_by_series.get(series_label or "", {})
    return baseline_avg.get((bench, metric, selector_fp))


def _metric_direction(row: dict[str, Any], metric: str) -> str:
    dir_map = row.get("directions") if isinstance(row.get("directions"), dict) else {}
    return str(dir_map.get(metric, "higher")).lower()


def _benchmark_subscore(
    items: list[dict[str, Any]],
) -> tuple[float, float] | None:
    """Benchmark subscore from width-aggregated raw values, with its present sub-weight.

    Aggregates the raw measurements across circuit widths first and normalizes the
    aggregate against the baseline, rather than normalizing each width separately
    and averaging the ratios. A single width whose baseline value sits at the noise
    floor otherwise produces an arbitrarily large ratio that dominates the composite.

    Returns None when no width has both a device value and a baseline value.
    """
    usable = [
        it
        for it in items
        if it["sub_weight"] > 0 and it["raw"] is not None and it["baseline"] is not None
    ]
    if not usable:
        return None
    device_total = sum(it["sub_weight"] * it["raw"] for it in usable)
    baseline_total = sum(it["sub_weight"] * it["baseline"] for it in usable)
    present_sub = sum(it["sub_weight"] for it in usable)
    lower_is_better = any(it["direction"] == "lower" for it in usable)
    if lower_is_better:
        if device_total <= 0:
            return None
        return 100.0 * baseline_total / device_total, present_sub
    if baseline_total <= 0:
        return None
    return 100.0 * device_total / baseline_total, present_sub


def _get_raw_metric_value(row: dict[str, Any], metric: str) -> float | None:
    """Return the raw metric value from row.results when present, else None."""
    results = row.get("results") if isinstance(row.get("results"), dict) else {}
    if metric not in results:
        return None
    return _coerce_float(results.get(metric))


class _TimestampedValue(NamedTuple):
    timestamp: datetime
    source_timestamp: Any
    value: Any


class _RawValue(NamedTuple):
    raw: float
    baseline: float | None
    direction: str


def _latest_timestamped_value(
    current: _TimestampedValue | None,
    row: dict[str, Any],
    value: Any,
) -> _TimestampedValue | None:
    """Keep a value when its row is newer than the current selection."""
    if value is None:
        return current
    timestamp = parse_timestamp(row.get("timestamp", ""))
    if timestamp is None or (current is not None and timestamp <= current.timestamp):
        return current
    return _TimestampedValue(timestamp, row.get("timestamp"), value)


def _record_outcome(row: dict[str, Any]) -> str | None:
    """Return the row's non-completed outcome, or None for a completed run.

    `etl.flatten_row` only stamps `outcome` for error / unsupported /
    not_applicable records, so an absent (or non-string) field means completed.
    """
    outcome = row.get("outcome")
    if isinstance(outcome, str) and outcome and outcome != "completed":
        return outcome
    return None


def _pick_latest_outcome_row(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Among outcome records for one benchmark instance, the latest claim wins."""
    picked = None
    picked_ts = None
    for cand in candidates:
        ts = parse_timestamp(cand.get("timestamp", ""))
        if ts is None:
            continue
        if picked is None or picked_ts is None or ts > picked_ts:
            picked = cand
            picked_ts = ts
    return picked


def _reported_outcome_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Fields stamped onto a component breakdown for the winning outcome record.

    Only the outcome, its human-readable reason and the claim's timestamp are
    surfaced; the full record (error_message, source, ...) stays available in
    benchmark.latest.json.
    """
    detail = row.get("outcome_detail") if isinstance(row.get("outcome_detail"), dict) else {}
    return {
        "reported_outcome": _record_outcome(row),
        "reported_outcome_reason": detail.get("reason"),
        "reported_outcome_timestamp": row.get("timestamp"),
    }


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
    row_major_by_id = {rid: _series_major(series_label) for rid, series_label in row_series.items()}
    component_cache: dict[str | None, list[dict[str, Any]]] = {}
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
        if picked_series not in component_cache:
            component_cache[picked_series] = _components_for_series(scoring_cfg, picked_series)
        components_flat = component_cache[picked_series]
        if not components_flat:
            # No components configured; skip this device
            continue
        breakdown: dict[str, dict[str, Any]] = {}
        numerator = 0.0
        sum_w_defined = 0.0
        groups: dict[tuple[Any, str], dict[str, Any]] = {}
        ungrouped: list[tuple[float, float]] = []

        for comp in components_flat:
            bench_field = comp.get("benchmark")
            allowed_names = _component_benchmarks(comp)
            metric = comp.get("metric")
            group_label = comp.get("_group_label")
            group_weight = float(comp.get("_group_weight", 1.0))
            sub_weight = float(comp.get("_sub_weight", 0.0))
            weight = float(comp.get("_effective_weight", _parse_weight(comp.get("weight", 0.0))))
            selector = comp.get("selector") if isinstance(comp.get("selector"), dict) else None
            # Prefer the primary benchmark name for label when available
            primary_bench = _primary_benchmark(comp)
            label = comp.get("label") or (
                f"{primary_bench}:{metric}" if primary_bench and metric else "component"
            )
            # Always include every component's weight in the denominator; if a
            # component is missing for this device, its normalized contribution is 0.
            sum_w_defined += weight

            selector_fp = _selector_fingerprint(selector)
            latest_normalized = None
            latest_raw = None
            # Non-completed outcome records (error / unsupported / not_applicable)
            # for this benchmark instance, and whether any completed record
            # exists for it (a completed record always supersedes outcomes).
            outcome_rows: list[dict[str, Any]] = []
            has_completed_record = False
            for r in rows:
                if allowed_names and derive_benchmark_name(r) not in allowed_names:
                    continue
                if not _component_param_matches(comp, r):
                    continue
                series_label = row_series.get(id(r))
                if picked_major is not None:
                    if row_major_by_id.get(id(r)) != picked_major:
                        continue
                elif series_label != picked_series:
                    continue
                if _record_outcome(r) is not None:
                    outcome_rows.append(r)
                    continue
                has_completed_record = True
                normalized_val = _get_normalized_metric_value(
                    r, metric, series_label, selector_fp, baseline_avg_by_series
                )
                is_baseline_row = _is_baseline_row_for_series(scoring_cfg, series_label, r)
                if normalized_val is not None and is_baseline_row:
                    # Keep baseline components anchored at 100 in platform composites.
                    normalized_val = 100.0
                latest_normalized = _latest_timestamped_value(
                    latest_normalized, r, normalized_val
                )
                raw_val = _get_raw_metric_value(r, metric)
                if raw_val is not None:
                    # Anchor the baseline device against its own value so its
                    # width-aggregated subscores come out at exactly 100.
                    base_val = (
                        raw_val
                        if is_baseline_row
                        else _get_baseline_metric_value(
                            r, metric, series_label, selector_fp, baseline_avg_by_series
                        )
                    )
                    latest_raw = _latest_timestamped_value(
                        latest_raw,
                        r,
                        _RawValue(raw_val, base_val, _metric_direction(r, metric)),
                    )

            normalized_ts = (
                latest_normalized.source_timestamp if latest_normalized else None
            )
            normalized_value = latest_normalized.value if latest_normalized else None
            raw_ts = latest_raw.source_timestamp if latest_raw else None
            raw_value, baseline_value, direction = (
                latest_raw.value if latest_raw else (None, None, "higher")
            )

            aggregation = comp.get("_group_aggregation", "arithmetic")
            item = {
                "label": label,
                "sub_weight": sub_weight,
                "weight": weight,
                "normalized": normalized_value,
                "raw": raw_value,
                "baseline": baseline_value,
                "direction": direction,
            }
            if group_label is not None:
                group = groups.setdefault(
                    (group_label, aggregation),
                    {
                        "group_weight": group_weight,
                        "aggregation": aggregation,
                        "items": [],
                    },
                )
                group["items"].append(item)
            elif normalized_value is not None:
                ungrouped.append((weight, normalized_value))

            breakdown[label] = {
                "metric": metric,
                "weight": weight,
                "group": group_label,
                "group_weight": group_weight,
                "sub_weight": sub_weight,
                "aggregation": aggregation,
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

            # Surface a structural qubit requirement so the UI can tell a
            # benchmark a device cannot run (device qubits below the
            # requirement) apart from one that is simply missing a submission.
            # Priority: an explicit required_num_qubits on the component config
            # (for components whose size lives in the metric name, like
            # EPLG-100), then a num_qubits selector, then a width selector
            # (Mirror Circuits, where width is the qubit count).
            required = comp.get("required_num_qubits")
            if not isinstance(required, int) and isinstance(selector, dict):
                for key in ("num_qubits", "width"):
                    if isinstance(selector.get(key), int):
                        required = selector[key]
                        break
            if isinstance(required, int):
                breakdown[label]["required_num_qubits"] = required

            # Surface the winning reported outcome (see README "Record
            # outcomes") so the UI can distinguish "the device cannot run
            # this" / "the attempt errored" from a plain missing submission.
            # A completed record for the instance supersedes every outcome
            # record; among outcome records the latest wins. Scoring above is
            # untouched: the component still contributes 0 to the numerator
            # and its weight stays in the denominator.
            if not has_completed_record:
                reported = _pick_latest_outcome_row(outcome_rows)
                if reported is not None:
                    breakdown[label].update(_reported_outcome_fields(reported))

        for weight, value in ungrouped:
            numerator += weight * value

        # Fold each group into the composite. Arithmetic groups normalize the
        # weighted raw aggregate once; harmonic groups combine normalized values.
        # Preserve the previous arithmetic-before-harmonic addition order so the
        # published floating-point scores stay byte-for-byte stable.
        ordered_groups = sorted(
            groups.values(), key=lambda group: group["aggregation"] == "harmonic"
        )
        for group in ordered_groups:
            items = group["items"]
            total_sub = sum(it["sub_weight"] for it in items if it["sub_weight"] > 0)
            if group["aggregation"] == "harmonic":
                present = [
                    (it["sub_weight"], it["normalized"])
                    for it in items
                    if it["sub_weight"] > 0 and it["normalized"] is not None
                ]
                present_sub = sum(weight for weight, _value in present)
                mean = _weighted_harmonic_mean(present)
                if total_sub > 0 and mean is not None:
                    numerator += group["group_weight"] * mean * (present_sub / total_sub)
                continue

            subscore = _benchmark_subscore(items) if total_sub > 0 else None
            if subscore is None:
                # No width has both a device and a baseline raw value (for example
                # derived metrics such as BSEQ that only carry normalized scores).
                for it in items:
                    if it["normalized"] is not None:
                        numerator += it["weight"] * it["normalized"]
                continue
            value, present_sub = subscore
            numerator += float(group["group_weight"]) * value * (present_sub / total_sub)
            for it in items:
                breakdown[it["label"]]["group_subscore"] = value

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
