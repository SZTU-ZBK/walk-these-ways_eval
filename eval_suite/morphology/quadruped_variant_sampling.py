"""Sample GenLoco-style random quadruped geometry/mass for offline Go1 URDF batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from eval_suite.morphology import quadruped_generator as qg

RANDOMIZED_SCALE_LOW = 0.8
RANDOMIZED_SCALE_HIGH = 1.2

GO1_LENGTH = 0.3762
GO1_HEIGHT = 0.34
NOMINAL_INIT_Z = 0.34
NOMINAL_LEG_SUM = float(2.0 * qg.go1_cylinder[1])

IMU_MASS_IDX = 0
LEG_NAMES: tuple[str, ...] = ("FR", "FL", "RR", "RL")

# Nominal hip offsets from go1.urdf.
NOM_HIP_X = 0.1881
NOM_HIP_Y = 0.04675
NOM_HIP_FIXED = 0.08


def leg_mass_indices(leg_index: int) -> tuple[int, int, int, int, int]:
    base = 1 + leg_index * 5
    return base, base + 1, base + 2, base + 3, base + 4


@dataclass
class QuadrupedVariantSample:
    randomized_scale: float
    x_base: float
    y_base: float
    z_base: float
    base_mass: float
    base_density: float
    hip_joint_link_pos_1: float
    hip_joint_link_pos_2: float
    hip_fixed_link_pos: float
    upper_cylinder_radius: np.ndarray
    upper_cylinder_length: np.ndarray
    lower_cylinder_radius: np.ndarray
    lower_cylinder_length: np.ndarray
    link_masses: np.ndarray
    delta_scale_used: float

    @property
    def upper_shoulder_y(self) -> float:
        return NOM_HIP_FIXED * (self.hip_fixed_link_pos / NOM_HIP_FIXED)

    def approx_footprint_x(self) -> float:
        return 2.0 * self.hip_joint_link_pos_1 + 2.0 * self.x_base

    def body_length_scale(self) -> float:
        return self.approx_footprint_x() / GO1_LENGTH

    def root_z_offset(self) -> float:
        leg_sum = float(np.mean(self.upper_cylinder_length)) + float(np.mean(self.lower_cylinder_length))
        stretch = leg_sum - NOMINAL_LEG_SUM
        return (NOMINAL_INIT_Z - GO1_HEIGHT) + 0.5 * stretch

    def to_manifest_dict(
        self,
        *,
        variant_index: int,
        urdf_relpath: str,
        usd_relpath: str | None = None,
    ) -> dict[str, Any]:
        return {
            "variant_index": variant_index,
            "urdf": urdf_relpath,
            "usd": usd_relpath,
            "body_length_scale": self.body_length_scale(),
            "root_z_offset": self.root_z_offset(),
            "approx_footprint_x": self.approx_footprint_x(),
            "randomized_scale": self.randomized_scale,
            "delta_scale": self.delta_scale_used,
            "x_base": self.x_base,
            "y_base": self.y_base,
            "z_base": self.z_base,
            "base_mass": self.base_mass,
            "base_density": self.base_density,
            "hip_joint_link_pos_1": self.hip_joint_link_pos_1,
            "hip_joint_link_pos_2": self.hip_joint_link_pos_2,
            "hip_fixed_link_pos": self.hip_fixed_link_pos,
            "upper_shoulder_y": self.upper_shoulder_y,
            "upper_cylinder_radius": self.upper_cylinder_radius.tolist(),
            "upper_cylinder_length": self.upper_cylinder_length.tolist(),
            "lower_cylinder_radius": self.lower_cylinder_radius.tolist(),
            "lower_cylinder_length": self.lower_cylinder_length.tolist(),
            "link_masses": self.link_masses.tolist(),
        }


def _sample_hip_offsets(rng: np.random.Generator, x_base: float, y_base: float) -> tuple[float, float, float]:
    hip_joint_link_pos_1 = float(NOM_HIP_X * (x_base / qg.go1_base[0]) * rng.uniform(0.9, 1.1))
    hip_joint_link_pos_2 = float(NOM_HIP_Y * (y_base / qg.go1_base[1]) * rng.uniform(0.9, 1.1))
    hip_fixed_link_pos = float(NOM_HIP_FIXED * rng.uniform(0.9, 1.1))
    return hip_joint_link_pos_1, hip_joint_link_pos_2, hip_fixed_link_pos


def sample_quadruped_variant(rng: np.random.Generator, delta_scale: float = 1.0) -> QuadrupedVariantSample:
    base_delta_cur = qg.base_delta * delta_scale
    density_delta_cur = qg.density_delta * delta_scale
    link_masses_delta_cur = qg.link_masses_delta * delta_scale
    cyl_delta_cur = qg.cyl_delta * delta_scale

    randomized_scale = float(rng.uniform(RANDOMIZED_SCALE_LOW, RANDOMIZED_SCALE_HIGH))
    base_low = (qg.go1_base - base_delta_cur) * randomized_scale
    base_high = (qg.go1_base + base_delta_cur) * randomized_scale
    x_base, y_base, z_base = rng.uniform(base_low, base_high).astype(float)

    cyl_low = (qg.go1_cylinder - cyl_delta_cur) * randomized_scale
    cyl_high = (qg.go1_cylinder + cyl_delta_cur) * randomized_scale

    lower_cylinder_radius = np.array([float(rng.uniform(cyl_low[0], cyl_high[0])) for _ in range(4)], dtype=np.float64)
    lower_cylinder_length = np.array([float(rng.uniform(cyl_low[1], cyl_high[1])) for _ in range(4)], dtype=np.float64)
    upper_cylinder_radius = np.array(
        [float(rng.uniform(cyl_low[0], cyl_high[0]) * rng.uniform(0.75, 1.25)) for _ in range(4)], dtype=np.float64
    )
    upper_cylinder_length = np.array(
        [float(rng.uniform(cyl_low[1], cyl_high[1]) * rng.uniform(0.75, 1.25)) for _ in range(4)], dtype=np.float64
    )

    link_masses = (
        rng.uniform(qg.link_masses - link_masses_delta_cur, qg.link_masses + link_masses_delta_cur)
        * (randomized_scale ** 3)
    ).astype(np.float64)
    base_density = float(rng.uniform(qg.go1_density - density_delta_cur, qg.go1_density + density_delta_cur))
    base_mass = float(base_density * x_base * y_base * z_base)

    hip_joint_link_pos_1, hip_joint_link_pos_2, hip_fixed_link_pos = _sample_hip_offsets(rng, x_base, y_base)

    return QuadrupedVariantSample(
        randomized_scale=randomized_scale,
        x_base=float(x_base),
        y_base=float(y_base),
        z_base=float(z_base),
        base_mass=base_mass,
        base_density=base_density,
        hip_joint_link_pos_1=hip_joint_link_pos_1,
        hip_joint_link_pos_2=hip_joint_link_pos_2,
        hip_fixed_link_pos=hip_fixed_link_pos,
        upper_cylinder_radius=upper_cylinder_radius,
        upper_cylinder_length=upper_cylinder_length,
        lower_cylinder_radius=lower_cylinder_radius,
        lower_cylinder_length=lower_cylinder_length,
        link_masses=link_masses,
        delta_scale_used=float(delta_scale),
    )


def sample_full_asymmetric_variant(rng: np.random.Generator, delta_scale: float = 0.5) -> QuadrupedVariantSample:
    base_delta_cur = qg.base_delta * delta_scale
    density_delta_cur = qg.density_delta * delta_scale
    link_masses_delta_cur = qg.link_masses_delta * delta_scale
    cyl_delta_cur = qg.cyl_delta * delta_scale

    randomized_scale = float(rng.uniform(RANDOMIZED_SCALE_LOW, RANDOMIZED_SCALE_HIGH))
    base_low = (qg.go1_base - base_delta_cur) * randomized_scale
    base_high = (qg.go1_base + base_delta_cur) * randomized_scale
    x_base, y_base, z_base = rng.uniform(base_low, base_high).astype(float)

    cyl_low = (qg.go1_cylinder - cyl_delta_cur) * randomized_scale
    cyl_high = (qg.go1_cylinder + cyl_delta_cur) * randomized_scale

    lower_cylinder_radius = np.array([float(rng.uniform(cyl_low[0], cyl_high[0])) for _ in range(4)], dtype=np.float64)
    lower_cylinder_length = np.array([float(rng.uniform(cyl_low[1], cyl_high[1])) for _ in range(4)], dtype=np.float64)
    upper_cylinder_radius = np.array(
        [float(rng.uniform(cyl_low[0], cyl_high[0]) * rng.uniform(0.75, 1.25)) for _ in range(4)], dtype=np.float64
    )
    upper_cylinder_length = np.array(
        [float(rng.uniform(cyl_low[1], cyl_high[1]) * rng.uniform(0.75, 1.25)) for _ in range(4)], dtype=np.float64
    )

    link_masses = np.zeros_like(qg.link_masses, dtype=np.float64)
    link_masses[0] = float(
        rng.uniform(qg.link_masses[0] - link_masses_delta_cur[0], qg.link_masses[0] + link_masses_delta_cur[0])
        * (randomized_scale ** 3)
    )
    for leg in range(4):
        mi = leg_mass_indices(leg)
        for idx in mi:
            link_masses[idx] = float(
                rng.uniform(
                    qg.link_masses[idx] - link_masses_delta_cur[idx],
                    qg.link_masses[idx] + link_masses_delta_cur[idx],
                )
                * (randomized_scale ** 3)
            )

    base_density = float(rng.uniform(qg.go1_density - density_delta_cur, qg.go1_density + density_delta_cur))
    base_mass = float(base_density * x_base * y_base * z_base)

    hip_joint_link_pos_1, hip_joint_link_pos_2, hip_fixed_link_pos = _sample_hip_offsets(rng, x_base, y_base)

    return QuadrupedVariantSample(
        randomized_scale=randomized_scale,
        x_base=float(x_base),
        y_base=float(y_base),
        z_base=float(z_base),
        base_mass=base_mass,
        base_density=base_density,
        hip_joint_link_pos_1=hip_joint_link_pos_1,
        hip_joint_link_pos_2=hip_joint_link_pos_2,
        hip_fixed_link_pos=hip_fixed_link_pos,
        upper_cylinder_radius=upper_cylinder_radius,
        upper_cylinder_length=upper_cylinder_length,
        lower_cylinder_radius=lower_cylinder_radius,
        lower_cylinder_length=lower_cylinder_length,
        link_masses=link_masses,
        delta_scale_used=float(delta_scale),
    )
