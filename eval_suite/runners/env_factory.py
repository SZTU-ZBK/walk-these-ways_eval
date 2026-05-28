"""Create Isaac Gym eval environments and load JIT policy."""

from __future__ import annotations

import glob
import pickle as pkl
from pathlib import Path
from typing import Any

import isaacgym  # noqa: F401 — must import before torch

assert isaacgym

from eval_suite.configs.eval_config import EvalConfig
from go1_gym import MINI_GYM_ROOT_DIR


def resolve_logdir(label: str, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(MINI_GYM_ROOT_DIR)
    dirs = glob.glob(str(root / "runs" / label / "*"))
    if not dirs:
        raise FileNotFoundError(f"No run directories found for label: {label}")
    return Path(sorted(dirs)[0])


def validate_checkpoint(logdir: Path) -> None:
    body = logdir / "checkpoints" / "body_latest.jit"
    adapt = logdir / "checkpoints" / "adaptation_module_latest.jit"
    missing = [p for p in (body, adapt) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint files: "
            + ", ".join(str(p) for p in missing)
            + ". Download pretrain checkpoint into runs/.../checkpoints/."
        )


def load_policy(logdir: Path):
    import torch

    body = torch.jit.load(str(logdir / "checkpoints" / "body_latest.jit"))
    adaptation_module = torch.jit.load(str(logdir / "checkpoints" / "adaptation_module_latest.jit"))

    def policy(obs, info=None):
        if info is None:
            info = {}
        latent = adaptation_module.forward(obs["obs_history"].to("cpu"))
        action = body.forward(torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1))
        info["latent"] = latent
        return action

    return policy


def _disable_domain_randomization(Cfg) -> None:
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_lag_timesteps = False


def _reset_cfg(Cfg) -> None:
    from go1_gym.envs.go1.go1_config import config_go1

    config_go1(Cfg)
    # legged_robot 运行时会写入 Cfg.command_ranges = dict，需在重建 env 前清掉
    if hasattr(Cfg, "command_ranges") and isinstance(getattr(Cfg, "command_ranges"), dict):
        delattr(Cfg, "command_ranges")


def _apply_checkpoint_cfg(Cfg, saved: dict) -> None:
    skip_keys = {"command_ranges"}
    for key, value in saved.items():
        if key in skip_keys or not hasattr(Cfg, key):
            continue
        section = getattr(Cfg, key)
        if isinstance(section, dict):
            continue
        for key2, value2 in value.items():
            setattr(section, key2, value2)


def make_eval_env(
    urdf_path: Path,
    cfg: EvalConfig,
    logdir: Path,
    root_z_offset: float = 0.0,
) -> Any:
    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    _reset_cfg(Cfg)

    with open(logdir / "parameters.pkl", "rb") as file:
        pkl_cfg = pkl.load(file)
        _apply_checkpoint_cfg(Cfg, pkl_cfg["Cfg"])

    _disable_domain_randomization(Cfg)
    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.control.control_type = "actuator_net"

    urdf_abs = Path(urdf_path)
    if not urdf_abs.is_absolute():
        urdf_abs = Path(MINI_GYM_ROOT_DIR) / urdf_abs
    Cfg.asset.file = str(urdf_abs)

    base_z = 0.34 + root_z_offset
    Cfg.init_state.pos = [0.0, 0.0, base_z]

    env = VelocityTrackingEasyEnv(sim_device=cfg.sim_device, headless=cfg.headless, cfg=Cfg)
    return HistoryWrapper(env)


def destroy_env(env) -> None:
    """Release Isaac Gym simulation resources between variant reloads."""
    target = env
    if hasattr(env, "env"):
        target = env.env
    if hasattr(target, "close"):
        try:
            target.close()
        except Exception:
            pass
    del env


def make_env_factory(
    urdf_path: Path,
    cfg: EvalConfig,
    logdir: Path,
    root_z_offset: float = 0.0,
):
    policy = load_policy(logdir)

    def factory():
        env = make_eval_env(urdf_path, cfg, logdir, root_z_offset=root_z_offset)
        return env, policy

    return factory
