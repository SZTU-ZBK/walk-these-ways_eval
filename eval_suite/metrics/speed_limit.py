"""Maximum forward velocity (robustness) via binary or linear search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval_suite.configs.eval_config import EvalConfig
from eval_suite.runners.rollout import RolloutResult, run_episode


@dataclass
class SpeedLimitResult:
    v_max: float
    search_log: list[dict[str, Any]] = field(default_factory=list)
    v_max_success_rate: float = 0.0


def _trial_success(result: RolloutResult, cfg: EvalConfig, vx: float) -> bool:
    if result.terminated_early:
        return False
    n = len(result.measured_vx)
    if n < cfg.speed_warmup_steps + cfg.speed_steady_steps:
        return False
    steady = result.measured_vx[-cfg.speed_steady_steps:]
    errors = np.abs(steady - vx)
    return float(np.mean(errors)) < cfg.speed_tracking_threshold


def evaluate_speed_limit(
    env: Any,
    policy: Any,
    cfg: EvalConfig,
) -> SpeedLimitResult:
    lo, hi = cfg.speed_min, cfg.speed_max
    log: list[dict[str, Any]] = []
    successes = 0
    trials = 0

    if cfg.speed_search_coarse_only:
        start = cfg.speed_min + cfg.speed_precision if cfg.speed_min == 0.0 else cfg.speed_min
        candidates = np.arange(start, cfg.speed_max + 1e-9, cfg.speed_precision)
        v_max = cfg.speed_min
        for vx in candidates:
            vx = float(vx)
            result = run_episode(
                env, policy, cfg,
                vx=vx, vy=0.0, wz=0.0,
                num_steps=cfg.speed_episode_steps,
            )
            ok = _trial_success(result, cfg, vx)
            trials += 1
            if ok:
                successes += 1
                v_max = vx
            log.append({"vx": vx, "success": ok, "terminated_early": result.terminated_early})
        rate = successes / max(trials, 1)
        return SpeedLimitResult(v_max=v_max, search_log=log, v_max_success_rate=rate)

    best = cfg.speed_min
    while hi - lo >= cfg.speed_precision:
        mid = round((lo + hi) / 2.0, 4)
        result = run_episode(
            env, policy, cfg,
            vx=mid, vy=0.0, wz=0.0,
            num_steps=cfg.speed_episode_steps,
        )
        ok = _trial_success(result, cfg, mid)
        trials += 1
        if ok:
            successes += 1
            best = mid
            lo = mid
        else:
            hi = mid
        log.append({"vx": mid, "success": ok, "terminated_early": result.terminated_early})

    rate = successes / max(trials, 1)
    return SpeedLimitResult(v_max=best, search_log=log, v_max_success_rate=rate)
