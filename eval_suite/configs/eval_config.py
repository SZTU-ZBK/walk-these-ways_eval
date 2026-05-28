"""Default evaluation hyperparameters (dataclass)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalConfig:
    logdir_label: str = "gait-conditioned-agility/pretrain-v0/train"
    sim_device: str = "cuda:0"
    headless: bool = True

    gait: list[float] = field(default_factory=lambda: [0.5, 0.0, 0.0])
    step_frequency: float = 3.0
    footswing_height: float = 0.08
    body_height_cmd: float = 0.0
    pitch_cmd: float = 0.0
    roll_cmd: float = 0.0
    stance_width_cmd: float = 0.25

    speed_min: float = 0.0
    speed_max: float = 4.0
    speed_precision: float = 0.2
    speed_tracking_threshold: float = 0.3
    speed_episode_steps: int = 250
    speed_warmup_steps: int = 50
    speed_steady_steps: int = 100

    yaw_test_vx: float = 1.5
    yaw_test_vy: float = 0.0
    # 偏航指标复用 stability 直线 episode（wz=0），以下字段保留兼容、不再单独 rollout
    yaw_test_wz: float = 0.0
    yaw_episode_steps: int = 250
    yaw_warmup_steps: int = 50

    stability_vx: float = 1.5
    stability_vy: float = 0.0
    stability_wz: float = 0.0
    stability_episode_steps: int = 250
    stability_warmup_steps: int = 50
    stability_steady_steps: int = 150

    sym_pool: Path = Path("eval_suite/assets/symmetric_v64")
    asym_pool: Path = Path("eval_suite/assets/full_asym_v64")
    baseline_urdf: Path = Path("resources/robots/go1/urdf/go1.urdf")

    num_variants: int = 16
    speed_search_coarse_only: bool = False
