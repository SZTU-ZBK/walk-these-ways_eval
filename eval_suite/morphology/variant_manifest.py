"""Load offline quadruped variant manifest.json for eval_suite."""

from __future__ import annotations

import json
from pathlib import Path


def load_manifest_entries(manifest_dir: Path) -> list[dict]:
    path = Path(manifest_dir).resolve() / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"manifest.json not found in {manifest_dir.resolve()}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("manifest.json must be a non-empty JSON array")
    return data


def manifest_severity(entry: dict) -> tuple[float, ...]:
    bs = float(entry.get("body_length_scale", 1.0))
    rz = float(entry.get("root_z_offset", 0.0))
    geo = abs(bs - 1.0) + abs(rz)
    tie = int(entry.get("variant_index", 0))
    ds = entry.get("delta_scale", None)
    if ds is not None:
        return (float(ds), geo, tie)
    return (geo, tie)


def sort_manifest_entries_by_severity(entries: list[dict]) -> list[dict]:
    indexed = sorted(enumerate(entries), key=lambda ie: (manifest_severity(ie[1]), ie[0]))
    out: list[dict] = []
    for new_idx, (_old_i, row) in enumerate(indexed):
        row = dict(row)
        row["variant_index"] = new_idx
        out.append(row)
    return out


def resolve_urdf_path(manifest_dir: Path, entry: dict) -> Path:
    root = Path(manifest_dir).resolve()
    rel = entry["urdf"]
    path = (root / rel).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF not found: {path}")
    return path
