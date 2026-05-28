"""Single-episode rollout with time-series logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from eval_suite.configs.eval_config import EvalConfig


@dataclass
class RolloutResult:
    t: np.ndarray
    measured_vx: np.ndarray
    measured_vy: np.ndarray
    measured_wz: np.ndarray
    cmd_vx: np.ndarray
    cmd_vy: np.ndarray
    cmd_wz: np.ndarray
    base_quat: np.ndarray
    roll: np.ndarray
    pitch: np.ndarray
    torques: np.ndarray
    dof_vel: np.ndarray
    power: np.ndarray
    robot_mass: float
    terminated_early: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _quat_to_roll_pitch(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def _set_commands(env, cfg: EvalConfig, vx: float, vy: float, wz: float) -> None:
    gait = torch.tensor(cfg.gait, device=env.commands.device, dtype=env.commands.dtype)
    env.commands[:, 0] = vx
    env.commands[:, 1] = vy
    env.commands[:, 2] = wz
    env.commands[:, 3] = cfg.body_height_cmd
    env.commands[:, 4] = cfg.step_frequency
    env.commands[:, 5:8] = gait
    env.commands[:, 8] = 0.5
    env.commands[:, 9] = cfg.footswing_height
    env.commands[:, 10] = cfg.pitch_cmd
    env.commands[:, 11] = cfg.roll_cmd
    env.commands[:, 12] = cfg.stance_width_cmd


def run_episode(
    env,
    policy,
    cfg: EvalConfig,
    *,
    vx: float,
    vy: float,
    wz: float,
    num_steps: int,
) -> RolloutResult:
    obs = env.reset()
    dt = float(env.dt)

    measured_vx = np.zeros(num_steps)
    measured_vy = np.zeros(num_steps)
    measured_wz = np.zeros(num_steps)
    cmd_vx = np.full(num_steps, vx)
    cmd_vy = np.full(num_steps, vy)
    cmd_wz = np.full(num_steps, wz)
    base_quat = np.zeros((num_steps, 4))
    torques = np.zeros((num_steps, env.num_dof))
    dof_vel = np.zeros((num_steps, env.num_dof))
    power = np.zeros(num_steps)
    terminated_early = False

    for i in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)
        _set_commands(env, cfg, vx, vy, wz)
        obs, rew, done, info = env.step(actions)

        measured_vx[i] = float(env.base_lin_vel[0, 0].cpu())
        measured_vy[i] = float(env.base_lin_vel[0, 1].cpu())
        measured_wz[i] = float(env.base_ang_vel[0, 2].cpu())
        base_quat[i] = env.base_quat[0].cpu().numpy()
        torques[i] = env.torques[0].detach().cpu().numpy()
        dof_vel[i] = env.dof_vel[0].detach().cpu().numpy()
        power[i] = float(torch.sum(torch.abs(env.torques[0] * env.dof_vel[0])).detach().cpu())

        if bool(done[0].item()):
            terminated_early = True
            measured_vx = measured_vx[: i + 1]
            measured_vy = measured_vy[: i + 1]
            measured_wz = measured_wz[: i + 1]
            cmd_vx = cmd_vx[: i + 1]
            cmd_vy = cmd_vy[: i + 1]
            cmd_wz = cmd_wz[: i + 1]
            base_quat = base_quat[: i + 1]
            torques = torques[: i + 1]
            dof_vel = dof_vel[: i + 1]
            power = power[: i + 1]
            break

    roll, pitch = _quat_to_roll_pitch(base_quat)
    n = len(measured_vx)
    t = np.arange(n) * dt
    inner = env.env if hasattr(env, "env") else env
    if hasattr(inner, "default_body_mass"):
        mass_val = inner.default_body_mass
        if hasattr(mass_val, "__getitem__"):
            mass_val = mass_val[0]
        mass = float(mass_val.cpu() if hasattr(mass_val, "cpu") else mass_val)
    else:
        mass = 4.8

    return RolloutResult(
        t=t,
        measured_vx=measured_vx,
        measured_vy=measured_vy,
        measured_wz=measured_wz,
        cmd_vx=cmd_vx,
        cmd_vy=cmd_vy,
        cmd_wz=cmd_wz,
        base_quat=base_quat,
        roll=roll,
        pitch=pitch,
        torques=torques,
        dof_vel=dof_vel,
        power=power,
        robot_mass=mass,
        terminated_early=terminated_early,
    )
