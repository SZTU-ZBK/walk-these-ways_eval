"""Straight-line yaw drift: heading offset and variance while walking forward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from eval_suite.configs.eval_config import EvalConfig
from eval_suite.runners.rollout import RolloutResult


@dataclass
class YawDriftResult:
    """Yaw accuracy on a straight-line command (vx>0, wz=0)."""

    yaw_offset_mean: float
    yaw_variance: float
    rollout: Any


def _quat_to_yaw(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def quat_yaw_series(quat: np.ndarray) -> np.ndarray:
    return _quat_to_yaw(quat)


def yaw_drift_series(yaw: np.ndarray, ref_idx: int) -> np.ndarray:
    return _wrap_to_pi(yaw - yaw[ref_idx])


def evaluate_yaw_drift(rollout: RolloutResult, cfg: EvalConfig) -> YawDriftResult:
    """Compute yaw offset / variance from a straight-line rollout (same episode as stability).

    - yaw_offset_mean: mean |yaw(t) - yaw_ref| in steady segment (rad)
    - yaw_variance: Var(yaw) in steady segment (rad²)
    - yaw_ref: heading at end of warmup (index stability_warmup_steps - 1)
    """
    yaw = _quat_to_yaw(rollout.base_quat)
    n = len(yaw)
    if n == 0:
        return YawDriftResult(yaw_offset_mean=0.0, yaw_variance=0.0, rollout=rollout)

    start = max(cfg.stability_warmup_steps, n - cfg.stability_steady_steps)
    steady_yaw = yaw[start:]
    ref_idx = min(max(cfg.stability_warmup_steps - 1, 0), n - 1)
    drift = _wrap_to_pi(steady_yaw - yaw[ref_idx])

    return YawDriftResult(
        yaw_offset_mean=float(np.mean(np.abs(drift))),
        yaw_variance=float(np.var(steady_yaw)),
        rollout=rollout,
    )
