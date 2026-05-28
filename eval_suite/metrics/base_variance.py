"""Base state variance during straight-line locomotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from eval_suite.configs.eval_config import EvalConfig
from eval_suite.runners.rollout import run_episode


@dataclass
class BaseVarianceResult:
    var_vx: float
    var_vy: float
    var_roll: float
    var_pitch: float
    base_variance_scalar: float
    rollout: Any


def evaluate_base_stability(
    env: Any,
    policy: Any,
    cfg: EvalConfig,
) -> BaseVarianceResult:
    result = run_episode(
        env, policy, cfg,
        vx=cfg.stability_vx,
        vy=cfg.stability_vy,
        wz=cfg.stability_wz,
        num_steps=cfg.stability_episode_steps,
    )

    start = max(cfg.stability_warmup_steps, len(result.measured_vx) - cfg.stability_steady_steps)
    vx = result.measured_vx[start:]
    vy = result.measured_vy[start:]
    roll = result.roll[start:]
    pitch = result.pitch[start:]

    var_vx = float(np.var(vx))
    var_vy = float(np.var(vy))
    var_roll = float(np.var(roll))
    var_pitch = float(np.var(pitch))
    scalar = float(np.mean([var_vx, var_vy, var_roll, var_pitch]))

    return BaseVarianceResult(
        var_vx=var_vx,
        var_vy=var_vy,
        var_roll=var_roll,
        var_pitch=var_pitch,
        base_variance_scalar=scalar,
        rollout=result,
    )
