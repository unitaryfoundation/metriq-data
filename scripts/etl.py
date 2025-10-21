from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional


# ---------------------------- Utilities ----------------------------

def iso_utc_now() -> str:
    dt = datetime.now(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def parse_timestamp(ts: str) -> Optional[datetime]:
    if not isinstance(ts, str):
        return None
    s = ts.strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def find_result_files(root: str) -> list[str]:
    pattern = os.path.join(root, "*/v*/**/results.json")
    seen = set()
    files: list[str] = []
    for p in glob.glob(pattern, recursive=True):
        if os.path.isfile(p):
            ap = os.path.abspath(p)
            if ap not in seen:
                seen.add(ap)
                files.append(ap)
    files.sort()
    return files


def path_version_segment(path: str) -> str:
    rel = os.path.relpath(path, os.getcwd())
    for seg in rel.split(os.sep):
        if seg.startswith("v"):
            return seg
    return "unknown"


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("app_version", "timestamp", "provider", "suite_id", "device", "job_type", "results", "params"):
        if k in row:
            out[k] = row[k]

    platform = row.get("platform")
    device_metadata = None
    if isinstance(platform, dict):
        device_metadata = platform.get("device_metadata")
        if "provider" not in out and isinstance(platform.get("provider"), str):
            out["provider"] = platform.get("provider")
        if "device" not in out and isinstance(platform.get("device"), str):
            out["device"] = platform.get("device")
    if device_metadata is None and "device_metadata" in row:
        device_metadata = row.get("device_metadata")
    if device_metadata is not None:
        out["device_metadata"] = device_metadata

    r = out.get("results") if "results" in out else row.get("results")
    if isinstance(r, dict):
        values = r.get("values") if isinstance(r.get("values"), dict) else None
        uncertainties = r.get("uncertainties") if isinstance(r.get("uncertainties"), dict) else None
        directions = r.get("directions") if isinstance(r.get("directions"), dict) else None
        if values is not None:
            out["results"] = values
            if uncertainties is not None and len(uncertainties) > 0:
                out["errors"] = uncertainties
            if directions is not None and len(directions) > 0:
                out["directions"] = directions
        else:
            out["results"] = r

    return out


def get_provider_device(row: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    provider = row.get("provider")
    device = row.get("device")
    if not provider or not device:
        platform = row.get("platform")
        if isinstance(platform, dict):
            provider = provider or platform.get("provider")
            device = device or platform.get("device")
    return provider, device


# ---------------------------- Platform Registry ----------------------------

PlatformKey = tuple[str, str]


def upsert_platform(registry: dict[PlatformKey, dict[str, Any]], row: dict[str, Any], src_file: str, version: str) -> None:
    provider, device = get_provider_device(row)
    if not provider or not device:
        return

    ts_str = row.get("timestamp")
    ts = parse_timestamp(ts_str) if ts_str else None
    if ts is None:
        return

    platform_info = row.get("platform") if isinstance(row.get("platform"), dict) else {}
    device_metadata = platform_info.get("device_metadata") if isinstance(platform_info, dict) else None
    if device_metadata is None:
        device_metadata = row.get("device_metadata")

    key: PlatformKey = (str(provider), str(device))
    entry = registry.get(key)
    ts_iso = ts.isoformat().replace("+00:00", "Z")

    if entry is None:
        entry = {
            "provider": provider,
            "device": device,
            "first_seen": ts_iso,
            "last_seen": ts_iso,
            "runs": 0,
            "metadata_history": {},
        }
        registry[key] = entry

    entry["runs"] = int(entry.get("runs", 0)) + 1

    def _cmp_parse(s: str) -> datetime:
        return parse_timestamp(s) or ts

    if _cmp_parse(entry["first_seen"]) > ts:
        entry["first_seen"] = ts_iso
    if _cmp_parse(entry["last_seen"]) < ts:
        entry["last_seen"] = ts_iso

    fp = canonical_json(device_metadata) if device_metadata is not None else "null"
    mh = entry["metadata_history"]
    hist = mh.get(fp)
    if hist is None:
        hist = {
            "device_metadata": device_metadata,
            "first_seen": ts_iso,
            "last_seen": ts_iso,
            "runs": 1,
        }
        mh[fp] = hist
    else:
        hist["runs"] = int(hist.get("runs", 0)) + 1
        first_seen_dt = parse_timestamp(hist["first_seen"])
        if first_seen_dt and first_seen_dt > ts:
            hist["first_seen"] = ts_iso
        last_seen_dt = parse_timestamp(hist["last_seen"])
        if last_seen_dt and last_seen_dt < ts:
            hist["last_seen"] = ts_iso


def _current_metadata_entry(entry: dict[str, Any]) -> dict[str, Any]:
    m_hist: dict[str, dict[str, Any]] = entry.get("metadata_history", {})
    if not m_hist:
        return {"device_metadata": None, "as_of": entry.get("last_seen")}
    platform_last = parse_timestamp(entry.get("last_seen", ""))
    best_item = None
    best_ts = None
    for hist in m_hist.values():
        h_last = parse_timestamp(hist.get("last_seen", ""))
        if platform_last and h_last and h_last == platform_last:
            return {"device_metadata": hist.get("device_metadata"), "as_of": hist.get("last_seen")}
        if h_last and (best_ts is None or h_last > best_ts):
            best_ts = h_last
            best_item = hist
    if best_item is None:
        return {"device_metadata": None, "as_of": entry.get("last_seen")}
    return {"device_metadata": best_item.get("device_metadata"), "as_of": best_item.get("last_seen")}


def write_platform_outputs(registry: dict[PlatformKey, dict[str, Any]], dist_path: str, generated_at: str) -> tuple[int, str]:
    platforms_dir = os.path.join(dist_path, "platforms")
    ensure_dir(platforms_dir)
    index_platforms: list[dict[str, Any]] = []

    for (provider, device) in sorted(registry.keys(), key=lambda k: (k[0], k[1])):
        entry = registry[(provider, device)]
        current = _current_metadata_entry(entry)
        provider_dir = os.path.join(platforms_dir, provider)
        ensure_dir(provider_dir)
        platform_file = os.path.join(provider_dir, f"{device}.json")

        history_list = list(entry.get("metadata_history", {}).values())
        history_list.sort(key=lambda h: parse_timestamp(h.get("first_seen", "")) or datetime.min.replace(tzinfo=timezone.utc))

        platform_payload = {
            "generated_at": generated_at,
            "provider": provider,
            "device": device,
            "first_seen": entry.get("first_seen"),
            "last_seen": entry.get("last_seen"),
            "runs": entry.get("runs", 0),
            "current": current,
            "history": history_list,
        }

        with open(platform_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(platform_payload, ensure_ascii=False, indent=2))
            f.write("\n")

        index_platforms.append({
            "provider": provider,
            "device": device,
            "first_seen": entry.get("first_seen"),
            "last_seen": entry.get("last_seen"),
            "runs": entry.get("runs", 0),
            "metadata_variants": len(entry.get("metadata_history", {})),
        })

    index_payload = {"generated_at": generated_at, "platforms": index_platforms}
    index_file = os.path.join(platforms_dir, "index.json")
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(index_payload, ensure_ascii=False, indent=2))
        f.write("\n")

    return len(index_platforms), platforms_dir


def derive_benchmark_name(r: dict[str, Any]) -> str:
    params = r.get("params") or {}
    if isinstance(params, dict) and isinstance(params.get("benchmark_name"), str):
        return params.get("benchmark_name")
    if isinstance(r.get("job_type"), str):
        return r.get("job_type")
    return "unknown"


def collect_flat_rows_and_registry(root: str, files: list[str]) -> tuple[
    list[dict[str, Any]], dict[int, str], dict[PlatformKey, dict[str, Any]]
]:
    registry: dict[PlatformKey, dict[str, Any]] = {}
    flat_rows: list[dict[str, Any]] = []
    row_series: dict[int, str] = {}

    for src in files:
        version = path_version_segment(src)
        try:
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load {src}: {e}", file=sys.stderr)
            continue

        if not isinstance(data, list):
            print(f"Warning: skipping {src} (not a list)", file=sys.stderr)
            continue

        for row in data:
            if not isinstance(row, dict):
                continue
            upsert_platform(registry, row, src, version)
            flat = flatten_row(row)
            if flat:
                flat_rows.append(flat)
                row_series[id(flat)] = version

    return flat_rows, row_series, registry


def write_benchmark_latest(dist_path: str, rows: list[dict[str, Any]]) -> str:
    latest_file = os.path.join(dist_path, "benchmark.latest.json")
    with open(latest_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(rows, ensure_ascii=False, indent=2))
        f.write("\n")
    return latest_file
