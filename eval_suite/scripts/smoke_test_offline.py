#!/usr/bin/env python3
"""离线 smoke test：无需 Isaac Gym，校验 URDF 池、manifest 与 checkpoint。"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval_suite.configs import load_eval_config
from eval_suite.morphology.variant_manifest import load_manifest_entries, resolve_urdf_path

REQUIRED_JOINTS = {
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
}


def _check_pool(pool: Path, label: str, min_count: int) -> None:
    entries = load_manifest_entries(pool)
    assert len(entries) >= min_count, f"{label}: 期望 >={min_count} 个变体，实际 {len(entries)}"
    e0 = entries[0]
    urdf = resolve_urdf_path(pool, e0)
    text = urdf.read_text(encoding="utf-8")
    for j in REQUIRED_JOINTS:
        assert j in text, f"{label}: URDF 缺少关节 {j}（{urdf.name}）"
    print(f"[smoke] {label}: {len(entries)} 个变体，关节命名 OK（{urdf.name}）")


def _check_asym_diff(pool: Path) -> None:
    entries = load_manifest_entries(pool)
    leg_lens = [e["lower_cylinder_length"] for e in entries[:8]]
    unique = len({tuple(map(float, x)) for x in leg_lens})
    assert unique > 1, "full_asym 池：前 8 个变体腿部长度应互不相同"
    print(f"[smoke] full_asym: 前 8 个变体中有 {unique} 种不同腿部长度")


def main() -> None:
    cfg = load_eval_config()
    baseline = (_REPO / cfg.baseline_urdf).resolve()
    assert baseline.is_file(), f"baseline URDF 不存在: {baseline}"

    sym = (_REPO / cfg.sym_pool).resolve()
    asym = (_REPO / cfg.asym_pool).resolve()
    _check_pool(sym, "symmetric", cfg.num_variants)
    _check_pool(asym, "full_asym", cfg.num_variants)
    _check_asym_diff(asym)

    logdir_glob = list((_REPO / "runs" / cfg.logdir_label).glob("*"))
    assert logdir_glob, f"未找到 run 目录: {cfg.logdir_label}"
    logdir = sorted(logdir_glob)[0]
    for name in ("body_latest.jit", "adaptation_module_latest.jit", "parameters.pkl"):
        p = logdir / "checkpoints" / name if name.endswith(".jit") else logdir / name
        assert p.is_file(), f"缺少 checkpoint: {p}"

    print("[smoke] 离线检查全部通过")


if __name__ == "__main__":
    main()
