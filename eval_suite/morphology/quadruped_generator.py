"""Go1 nominal geometry and mass parameters for GenLoco-style variant sampling."""

import numpy as np

# Trunk half-extents (m) — from go1.urdf collision box 0.3762 x 0.0935 x 0.114.
go1_base = np.array([0.1881, 0.04675, 0.057])
base_delta = go1_base / 2

go1_base_mass = 4.8
go1_density = go1_base_mass / np.prod(go1_base * 2)
density_delta = go1_density / 2

# Per-link masses: IMU + 4 legs x (hip, shoulder, thigh, calf, foot).
link_masses = np.array([
    0.001,
    0.510299, 0.001, 0.898919, 0.158015, 0.06,
    0.510299, 0.001, 0.898919, 0.158015, 0.06,
    0.510299, 0.001, 0.898919, 0.158015, 0.06,
    0.510299, 0.001, 0.898919, 0.158015, 0.06,
])
link_masses_delta = link_masses / 2

# Leg segment proxy: (radius, length) — thigh/calf segment length 0.213 m each.
go1_cylinder = np.array([0.012, 0.213])
cyl_delta = go1_cylinder / 2

RANDOMIZED_SCALE_LOW, RANDOMIZED_SCALE_HIGH = 0.8, 1.2

# Aliases used by ported sampling code (same names as gen_loco).
a1_base = go1_base
a1_base_mass = go1_base_mass
a1_density = go1_density
a1_cylinder = go1_cylinder
