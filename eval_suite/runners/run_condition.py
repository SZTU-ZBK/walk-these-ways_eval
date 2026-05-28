"""运行单一形态学评估工况。"""

from __future__ import annotations

import isaacgym  # noqa: F401 — must import before torch (via rollout)

assert isaacgym

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eval_suite.configs.eval_config import EvalConfig
from eval_suite.configs import load_eval_config
from eval_suite.metrics.aggregator import write_summary_csv, write_summary_json
from eval_suite.metrics.base_variance import evaluate_base_stability
from eval_suite.metrics.power import compute_power_from_rollout
from eval_suite.metrics.speed_limit import evaluate_speed_limit
from eval_suite.metrics.yaw_error import evaluate_yaw_drift, quat_yaw_series, yaw_drift_series
from eval_suite.morphology.variant_manifest import load_manifest_entries, resolve_urdf_path
from eval_suite.runners.env_factory import make_env_factory, resolve_logdir, validate_checkpoint
from go1_gym import MINI_GYM_ROOT_DIR


def _save_rollout_npz(path: Path, rollout, extra: dict[str, np.ndarray] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "t": rollout.t,
        "measured_vx": rollout.measured_vx,
        "measured_vy": rollout.measured_vy,
        "measured_wz": rollout.measured_wz,
        "cmd_vx": rollout.cmd_vx,
        "cmd_vy": rollout.cmd_vy,
        "cmd_wz": rollout.cmd_wz,
        "base_quat": rollout.base_quat,
        "roll": rollout.roll,
        "pitch": rollout.pitch,
        "torques": rollout.torques,
        "dof_vel": rollout.dof_vel,
        "power": rollout.power,
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)


def _variant_entries(condition: str, cfg: EvalConfig, repo_root: Path) -> list[dict[str, Any]]:
    if condition == "baseline":
        urdf = (repo_root / cfg.baseline_urdf).resolve()
        return [{
            "variant_index": 0,
            "urdf": str(cfg.baseline_urdf),
            "urdf_abs": str(urdf),
            "randomized_scale": 1.0,
            "root_z_offset": 0.0,
        }]
    pool = repo_root / (cfg.sym_pool if condition == "symmetric" else cfg.asym_pool)
    entries = load_manifest_entries(pool)[: cfg.num_variants]
    out = []
    for e in entries:
        urdf_abs = resolve_urdf_path(pool, e)
        row = dict(e)
        row["urdf_abs"] = str(urdf_abs)
        out.append(row)
    return out


def run_condition(
    condition: str,
    out_dir: Path,
    cfg: EvalConfig | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    repo_root = repo_root or Path(MINI_GYM_ROOT_DIR)
    cfg = cfg or load_eval_config()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logdir = resolve_logdir(cfg.logdir_label, repo_root)
    validate_checkpoint(logdir)

    entries = _variant_entries(condition, cfg, repo_root)
    manifest_copy = out_dir / "manifest.json"
    manifest_copy.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    (out_dir / "config.yaml").write_text(yaml.safe_dump({
        "condition": condition,
        "logdir": str(logdir),
        **{k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()},
    }), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    episodes_dir = out_dir / "episodes"

    for entry in entries:
        vid = int(entry["variant_index"])
        urdf_path = Path(entry["urdf_abs"])
        root_z = float(entry.get("root_z_offset", 0.0))
        factory = make_env_factory(urdf_path, cfg, logdir, root_z_offset=root_z)
        ep_dir = episodes_dir / f"variant_{vid:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        env, policy = factory()
        try:
            speed = evaluate_speed_limit(env, policy, cfg)
            stability = evaluate_base_stability(env, policy, cfg)
            yaw = evaluate_yaw_drift(stability.rollout, cfg)
        finally:
            from eval_suite.runners.env_factory import destroy_env
            destroy_env(env)

        (ep_dir / "speed_limit.json").write_text(json.dumps(speed.search_log, indent=2), encoding="utf-8")

        yaw_arr = quat_yaw_series(stability.rollout.base_quat)
        ref_idx = min(max(cfg.stability_warmup_steps - 1, 0), len(yaw_arr) - 1)
        _save_rollout_npz(
            ep_dir / "straight_line.npz",
            stability.rollout,
            extra={
                "yaw": yaw_arr,
                "yaw_drift": yaw_drift_series(yaw_arr, ref_idx),
            },
        )

        _save_rollout_npz(ep_dir / "stability.npz", stability.rollout)
        _save_rollout_npz(ep_dir / "power.npz", stability.rollout)

        power = compute_power_from_rollout(stability.rollout, cfg)

        row = {
            "condition": condition,
            "variant_index": vid,
            "randomized_scale": float(entry.get("randomized_scale", 1.0)),
            "v_max": speed.v_max,
            "v_max_success_rate": speed.v_max_success_rate,
            "yaw_offset_mean": yaw.yaw_offset_mean,
            "yaw_variance": yaw.yaw_variance,
            "var_vx": stability.var_vx,
            "var_vy": stability.var_vy,
            "var_roll": stability.var_roll,
            "var_pitch": stability.var_pitch,
            "base_variance_scalar": stability.base_variance_scalar,
            "mean_power": power.mean_power,
            "cot": power.cot,
            "urdf_path": str(urdf_path),
        }
        rows.append(row)
        print(
            f"[{condition}][variant_{vid:03d}] v_max={speed.v_max:.2f} "
            f"yaw_off={yaw.yaw_offset_mean:.4f} yaw_var={yaw.yaw_variance:.6f} "
            f"base_var={stability.base_variance_scalar:.6f} "
            f"power={power.mean_power:.2f}"
        )

    write_summary_csv(out_dir / "summary.csv", rows)
    write_summary_json(out_dir / "summary.json", rows)
    _write_report(out_dir, condition, rows)
    return rows


def _write_report(out_dir: Path, condition: str, rows: list[dict[str, Any]]) -> None:
    from eval_suite.metrics.aggregator import aggregate_rows

    agg = aggregate_rows(rows)
    lines = [
        f"# 形态学评估报告 — {condition}",
        "",
        f"生成时间: {datetime.now().isoformat()}",
        "",
        "## 汇总统计",
        "",
    ]
    for k, v in sorted(agg.items()):
        lines.append(f"- **{k}**: {v:.6g}" if isinstance(v, float) else f"- **{k}**: {v}")
    lines.extend(["", "## 各变体明细", ""])
    for r in rows:
        lines.append(
            f"- variant_{r['variant_index']:03d}: v_max={r['v_max']:.2f}, "
            f"yaw_off={r['yaw_offset_mean']:.4f}, yaw_var={r['yaw_variance']:.6f}, "
            f"base_var={r['base_variance_scalar']:.6f}, "
            f"power={r['mean_power']:.2f}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["baseline", "symmetric", "full_asym"], required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--num_variants", type=int, default=None, help="Override eval num_variants (default from config)")
    args = parser.parse_args()
    cfg = load_eval_config(args.config)
    if args.num_variants is not None:
        cfg.num_variants = args.num_variants
    run_condition(args.condition, args.out_dir, cfg=cfg)
