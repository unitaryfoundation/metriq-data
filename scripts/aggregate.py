#!/usr/bin/env python3
"""
aggregate.py — ETL for metriq-data

Scans all v*/<provider>/results.json files recursively, aggregates rows,
and writes outputs to:
  - dist/benchmark.latest.json
  - dist/platforms/index.json
  - dist/platforms/<provider>/<device>.json

Only uses Python standard library.
"""

from __future__ import annotations

import json
import os
import sys
import glob
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------- Utilities ----------------------------

def iso_utc_now() -> str:
    dt = datetime.now(timezone.utc)
    # Emit with trailing Z for clarity
    return dt.isoformat().replace("+00:00", "Z")


def parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse ISO timestamp into aware datetime (UTC on missing tz)."""
    if not isinstance(ts, str):
        return None
    s = ts.strip()
    if not s:
        return None
    try:
        # Support Z suffix
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
    """Return a canonical minified JSON string for stable hashing/fingerprints."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def find_result_files(root: str) -> List[str]:
    """Find all results.json files under source/version/provider layout.

    Pattern scanned:
      - */v*/**/results.json (preferred: source/version/provider/results.json)

    Returns absolute paths (for stable printing), sorted and de-duplicated.
    """
    pattern = os.path.join(root, "*/v*/**/results.json")
    seen = set()
    files: List[str] = []
    for p in glob.glob(pattern, recursive=True):
        if os.path.isfile(p):
            ap = os.path.abspath(p)
            if ap not in seen:
                seen.add(ap)
                files.append(ap)
    files.sort()
    return files


def path_version_segment(path: str) -> str:
    """Extract the nearest path segment starting with 'v' as version tag.

    Falls back to 'unknown' if none is found.
    """
    rel = os.path.relpath(path, os.getcwd())
    for seg in rel.split(os.sep):
        if seg.startswith("v"):
            return seg
    return "unknown"


def flatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a result row by lifting device_metadata to top-level and removing platform.

    Keeps commonly used top-level fields and preserves nested 'results' and 'params'.
    """
    out: Dict[str, Any] = {}
    for k in ("app_version", "timestamp", "provider", "suite_id", "device", "job_type", "results", "params"):
        if k in row:
            out[k] = row[k]

    # Prefer platform.device_metadata if present
    platform = row.get("platform")
    device_metadata = None
    if isinstance(platform, dict):
        device_metadata = platform.get("device_metadata")
        # Fill missing provider/device from platform if absent
        if "provider" not in out and isinstance(platform.get("provider"), str):
            out["provider"] = platform.get("provider")
        if "device" not in out and isinstance(platform.get("device"), str):
            out["device"] = platform.get("device")
    if device_metadata is None and "device_metadata" in row:
        device_metadata = row.get("device_metadata")
    if device_metadata is not None:
        out["device_metadata"] = device_metadata

    return out


def get_provider_device(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    provider = row.get("provider")
    device = row.get("device")
    if not provider or not device:
        platform = row.get("platform")
        if isinstance(platform, dict):
            provider = provider or platform.get("provider")
            device = device or platform.get("device")
    return provider, device


# ---------------------------- Platform Registry ----------------------------

PlatformKey = Tuple[str, str]


def upsert_platform(registry: Dict[PlatformKey, Dict[str, Any]], row: Dict[str, Any], src_file: str, version: str) -> None:
    """Update/insert platform summary from a single row.

    Tracks first_seen, last_seen, total runs, and unique device_metadata snapshots.
    """
    provider, device = get_provider_device(row)
    if not provider or not device:
        return  # Cannot attribute to a platform

    ts_str = row.get("timestamp")
    ts = parse_timestamp(ts_str) if ts_str else None
    if ts is None:
        # Skip rows without a valid timestamp for stable first/last seen tracking
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
            # metadata_history keyed by canonical fingerprint
            "metadata_history": {},
        }
        registry[key] = entry

    # Update platform-level stats
    entry["runs"] = int(entry.get("runs", 0)) + 1

    # Update first/last seen
    def _cmp_parse(s: str) -> datetime:
        return parse_timestamp(s) or ts

    if _cmp_parse(entry["first_seen"]) > ts:
        entry["first_seen"] = ts_iso
    if _cmp_parse(entry["last_seen"]) < ts:
        entry["last_seen"] = ts_iso

    # Track metadata snapshots
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
        # Update metahistory first/last seen
        if parse_timestamp(hist["first_seen"]) and parse_timestamp(hist["first_seen"]) > ts:
            hist["first_seen"] = ts_iso
        if parse_timestamp(hist["last_seen"]) and parse_timestamp(hist["last_seen"]) < ts:
            hist["last_seen"] = ts_iso


def _current_metadata_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the metadata history item corresponding to platform last_seen."""
    m_hist: Dict[str, Dict[str, Any]] = entry.get("metadata_history", {})
    if not m_hist:
        return {"device_metadata": None, "as_of": entry.get("last_seen")}
    # Choose the history item whose last_seen matches platform last_seen; if none, pick the most recent
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


def write_platform_outputs(registry: Dict[PlatformKey, Dict[str, Any]], dist_path: str, generated_at: str) -> Tuple[int, str]:
    """Write dist/platforms/index.json and per-platform JSON files.

    Returns a tuple of (platform_count, platforms_dir_path)
    """
    platforms_dir = os.path.join(dist_path, "platforms")
    ensure_dir(platforms_dir)

    # Summaries for index
    index_platforms: List[Dict[str, Any]] = []

    # Deterministic ordering
    for (provider, device) in sorted(registry.keys(), key=lambda k: (k[0], k[1])):
        entry = registry[(provider, device)]
        current = _current_metadata_entry(entry)

        # Write per-platform file
        provider_dir = os.path.join(platforms_dir, provider)
        ensure_dir(provider_dir)
        platform_file = os.path.join(provider_dir, f"{device}.json")

        history_list = list(entry.get("metadata_history", {}).values())
        # Sort history by first_seen ascending for readability
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
            f.write(json.dumps(platform_payload, separators=(",", ":"), ensure_ascii=False))

        # Build index entry
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
        f.write(json.dumps(index_payload, separators=(",", ":"), ensure_ascii=False))

    return len(index_platforms), platforms_dir


# ---------------------------- Main ETL ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    dist_path = os.path.join(root, "dist")
    ensure_dir(dist_path)

    generated_at = iso_utc_now()

    files = find_result_files(root)
    total_files = len(files)
    registry: Dict[PlatformKey, Dict[str, Any]] = {}
    flat_rows: List[Dict[str, Any]] = []

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
            # Track platform registry
            upsert_platform(registry, row, src, version)

            # Build flattened row
            flat = flatten_row(row)
            if flat:
                flat_rows.append(flat)

    # Write benchmark.latest.json (list of flattened rows)
    latest_file = os.path.join(dist_path, "benchmark.latest.json")
    with open(latest_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(flat_rows, separators=(",", ":"), ensure_ascii=False))

    # Write platforms outputs
    platform_count, platforms_dir = write_platform_outputs(registry, dist_path, generated_at)

    print(
        f"Processed {len(flat_rows)} rows across {platform_count} platforms from {total_files} files."
    )
    print(f"Wrote: {os.path.relpath(latest_file, root)} and {os.path.relpath(platforms_dir, root)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
