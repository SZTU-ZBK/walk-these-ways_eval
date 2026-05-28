"""Mechanical power consumption."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval_suite.configs.eval_config import EvalConfig
from eval_suite.runners.rollout import RolloutResult


@dataclass
class PowerResult:
    mean_power: float
    cot: float


def compute_power_from_rollout(rollout: RolloutResult, cfg: EvalConfig) -> PowerResult:
    start = max(cfg.stability_warmup_steps, len(rollout.power) - cfg.stability_steady_steps)
    steady_power = rollout.power[start:]
    mean_power = float(np.mean(steady_power))

    vx = rollout.measured_vx[start:]
    m = rollout.robot_mass
    g = 9.8
    v = np.maximum(np.abs(vx), 1e-3)
    cot = float(mean_power / (m * g * float(np.mean(v))))
    return PowerResult(mean_power=mean_power, cot=cot)
