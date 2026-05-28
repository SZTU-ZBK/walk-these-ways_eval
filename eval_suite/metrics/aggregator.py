"""Aggregate and persist evaluation summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "condition",
    "variant_index",
    "randomized_scale",
    "v_max",
    "v_max_success_rate",
    "yaw_offset_mean",
    "yaw_variance",
    "var_vx",
    "var_vy",
    "var_roll",
    "var_pitch",
    "base_variance_scalar",
    "mean_power",
    "cot",
    "urdf_path",
]


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    numeric_keys = [k for k in SUMMARY_COLUMNS if k not in ("condition", "urdf_path")]
    agg: dict[str, Any] = {"count": len(rows)}
    for key in numeric_keys:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if vals:
            agg[f"{key}_mean"] = sum(vals) / len(vals)
            agg[f"{key}_std"] = (sum((v - agg[f"{key}_mean"]) ** 2 for v in vals) / len(vals)) ** 0.5
    return agg


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_json(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = {"rows": rows, "aggregate": aggregate_rows(rows)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
