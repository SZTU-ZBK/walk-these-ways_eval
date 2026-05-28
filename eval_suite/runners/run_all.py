#!/usr/bin/env python3
"""运行三种形态学评估工况。"""

from __future__ import annotations

import isaacgym  # noqa: F401

assert isaacgym

import argparse
from datetime import datetime
from pathlib import Path

from eval_suite.configs import load_eval_config
from eval_suite.metrics.aggregator import aggregate_rows, write_summary_csv, write_summary_json
from eval_suite.scripts.check_prerequisites import check_all
from eval_suite.runners.run_condition import run_condition
from go1_gym import MINI_GYM_ROOT_DIR


def _write_combined_report(out_root: Path, all_rows: list[dict]) -> None:
    lines = [
        "# 形态学评估 — 综合对比报告",
        "",
        f"结果根目录: `{out_root}`",
        "",
        "## 工况对比",
        "",
        "| 工况 | v_max 均值 | yaw 偏移均值 (rad) | yaw 方差均值 (rad²) | base 方差均值 | 功率均值 |",
        "|------|------------|--------------------|---------------------|---------------|----------|",
    ]
    for cond in ("baseline", "symmetric", "full_asym"):
        subset = [r for r in all_rows if r["condition"] == cond]
        if not subset:
            continue
        agg = aggregate_rows(subset)
        lines.append(
            f"| {cond} | {agg.get('v_max_mean', 0):.3f} | "
            f"{agg.get('yaw_offset_mean_mean', 0):.4f} | "
            f"{agg.get('yaw_variance_mean', 0):.6f} | "
            f"{agg.get('base_variance_scalar_mean', 0):.6f} | "
            f"{agg.get('mean_power_mean', 0):.2f} |"
        )
    (out_root / "COMBINED_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=str, default=None, help="Run label under runs/")
    parser.add_argument("--sym_pool", type=Path, default=None)
    parser.add_argument("--asym_pool", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--conditions", nargs="+", default=["baseline", "symmetric", "full_asym"])
    parser.add_argument("--speed_search_coarse_only", action="store_true")
    parser.add_argument("--num_variants", type=int, default=None, help="Override eval num_variants (default 16)")
    args = parser.parse_args()

    check_all(Path(MINI_GYM_ROOT_DIR))

    cfg = load_eval_config(args.config)
    if args.logdir:
        cfg.logdir_label = args.logdir
    if args.sym_pool:
        cfg.sym_pool = args.sym_pool
    if args.asym_pool:
        cfg.asym_pool = args.asym_pool
    cfg.headless = args.headless
    if args.speed_search_coarse_only:
        cfg.speed_search_coarse_only = True
    if args.num_variants is not None:
        cfg.num_variants = args.num_variants

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_dir or Path(MINI_GYM_ROOT_DIR) / "eval_suite" / "results" / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for condition in args.conditions:
        cond_dir = out_root / condition
        rows = run_condition(condition, cond_dir, cfg=cfg)
        all_rows.extend(rows)

    write_summary_csv(out_root / "summary_all.csv", all_rows)
    write_summary_json(out_root / "summary_all.json", all_rows)
    _write_combined_report(out_root, all_rows)
    print(f"[run_all] 评估完成，结果目录: {out_root}")


if __name__ == "__main__":
    main()
