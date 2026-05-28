"""Write GenLoco-style randomized Go1 URDF with Go1 joint/link naming."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from eval_suite.morphology.quadruped_generator import go1_base_mass
from eval_suite.morphology.quadruped_variant_sampling import QuadrupedVariantSample, leg_mass_indices

NOM_LEG_LEN = 0.213
NOM_HIP_CYL_LEN = 0.04
NOM_HIP_CYL_R = 0.046
NOM_TOE_COLL_RADIUS = 0.01

HIP_LIMIT = ("33.5", "-0.802851455917", "0.802851455917", "50")
THIGH_LIMIT = ("33.5", "-1.0471975512", "4.18879020479", "28")
CALF_LIMIT = ("33.5", "-2.69653369433", "-0.916297857297", "28")


def _meshes_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "resources" / "robots" / "go1" / "meshes"


def _mesh_prefix(urdf_path: Path) -> str:
    return Path(os.path.relpath(_meshes_dir().resolve(), urdf_path.parent.resolve())).as_posix()


def _fmt(x: float) -> str:
    return f"{x:.10g}"


def _inertial(link, xyz, mass, ixx, ixy, ixz, iyy, iyz, izz):
    ine = ET.SubElement(link, "inertial")
    o = ET.SubElement(ine, "origin")
    o.set("rpy", "0 0 0")
    o.set("xyz", f"{_fmt(xyz[0])} {_fmt(xyz[1])} {_fmt(xyz[2])}")
    ET.SubElement(ine, "mass").set("value", _fmt(mass))
    it = ET.SubElement(ine, "inertia")
    it.set("ixx", _fmt(ixx))
    it.set("ixy", _fmt(ixy))
    it.set("ixz", _fmt(ixz))
    it.set("iyy", _fmt(iyy))
    it.set("iyz", _fmt(iyz))
    it.set("izz", _fmt(izz))


def _scale_I(ixx, ixy, ixz, iyy, iyz, izz, m_nom, m_new, g=1.0):
    s = (m_new / max(m_nom, 1e-12)) * (g ** 2)
    return ixx * s, ixy * s, ixz * s, iyy * s, iyz * s, izz * s


def _materials(robot):
    for name, rgba in [
        ("black", "0.0 0.0 0.0 1.0"),
        ("silver", "0.913725490196 0.913725490196 0.847058823529 1.0"),
        ("red", "0.8 0.0 0.0 1.0"),
    ]:
        m = ET.SubElement(robot, "material")
        m.set("name", name)
        ET.SubElement(m, "color").set("rgba", rgba)


def _revolute(robot, name, parent, child, xyz, axis, effort, lo, hi, vel):
    j = ET.SubElement(robot, "joint")
    j.set("name", name)
    j.set("type", "revolute")
    o = ET.SubElement(j, "origin")
    o.set("rpy", "0 0 0")
    o.set("xyz", f"{_fmt(xyz[0])} {_fmt(xyz[1])} {_fmt(xyz[2])}")
    ET.SubElement(j, "parent").set("link", parent)
    ET.SubElement(j, "child").set("link", child)
    ET.SubElement(j, "axis").set("xyz", axis)
    d = ET.SubElement(j, "dynamics")
    d.set("damping", "0")
    d.set("friction", "0")
    lim = ET.SubElement(j, "limit")
    lim.set("effort", effort)
    lim.set("lower", lo)
    lim.set("upper", hi)
    lim.set("velocity", vel)


def _fixed(robot, name, parent, child, xyz, dont_collapse=False):
    j = ET.SubElement(robot, "joint")
    j.set("name", name)
    j.set("type", "fixed")
    if dont_collapse:
        j.set("dont_collapse", "true")
    o = ET.SubElement(j, "origin")
    o.set("rpy", "0 0 0")
    o.set("xyz", f"{_fmt(xyz[0])} {_fmt(xyz[1])} {_fmt(xyz[2])}")
    ET.SubElement(j, "parent").set("link", parent)
    ET.SubElement(j, "child").set("link", child)


@dataclass(frozen=True)
class _LegLayout:
    prefix: str
    hip_sign: tuple[int, int]
    shoulder_sign: int
    hip_vis_rpy: str
    thigh_mesh: str
    hip_com: tuple[float, float, float]
    hip_i_off: tuple[float, float, float]
    thigh_i_off: tuple[float, float, float]


def _leg_layouts():
    return [
        _LegLayout("FR", (1, -1), -1, "3.14159265359 0 0", "thigh_mirror.stl",
                   (-0.00541, 0.00074, 6e-06), (7.788013e-06, 2.2016e-07, 1.7175e-08),
                   (-1.02809e-07, 0.000337529085, 5.816563e-06)),
        _LegLayout("FL", (1, 1), 1, "0 0 0", "thigh.stl",
                   (-0.00541, -0.00074, 6e-06), (-7.788013e-06, -2.2016e-07, -1.7175e-08),
                   (1.02809e-07, 0.000337529085, -5.816563e-06)),
        _LegLayout("RR", (-1, -1), -1, "3.14159265359 3.14159265359 0", "thigh_mirror.stl",
                   (0.00541, -0.00074, 6e-06), (-7.788013e-06, 2.2016e-07, 1.7175e-08),
                   (-1.02809e-07, 0.000337529085, 5.816563e-06)),
        _LegLayout("RL", (-1, 1), 1, "0 3.14159265359 0", "thigh.stl",
                   (0.00541, 0.00074, 6e-06), (7.788013e-06, -2.2016e-07, -1.7175e-08),
                   (1.02809e-07, 0.000337529085, -5.816563e-06)),
    ]


def _add_leg(robot, sample, mesh_p, layout, leg_index, ul, ur):
    p = layout.prefix
    sx, sy = layout.hip_sign
    h1 = sample.hip_joint_link_pos_1
    h2 = sample.hip_joint_link_pos_2
    hf = sample.hip_fixed_link_pos
    ll = float(sample.lower_cylinder_length[leg_index])
    lr = float(sample.lower_cylinder_radius[leg_index])
    thigh_scale = ul / NOM_LEG_LEN

    mi = leg_mass_indices(leg_index)
    m_hip, m_sh, m_th, m_ca, m_foot = (float(sample.link_masses[i]) for i in mi)

    hip_xyz = (sx * h1, sy * h2, 0.0)
    sh_xyz = (0.0, layout.shoulder_sign * hf, 0.0)

    _revolute(robot, f"{p}_hip_joint", "trunk", f"{p}_hip", hip_xyz, "1 0 0", *HIP_LIMIT)

    hip = ET.SubElement(robot, "link")
    hip.set("name", f"{p}_hip")
    hv = ET.SubElement(hip, "visual")
    ho = ET.SubElement(hv, "origin")
    ho.set("rpy", layout.hip_vis_rpy)
    ho.set("xyz", "0 0 0")
    hg = ET.SubElement(hv, "geometry")
    hm = ET.SubElement(hg, "mesh")
    hm.set("filename", f"{mesh_p}/hip.stl")
    hm.set("scale", "1 1 1")
    ET.SubElement(hv, "material").set("name", "silver")
    hc = ET.SubElement(hip, "collision")
    hco = ET.SubElement(hc, "origin")
    hco.set("rpy", "1.57079632679 0 0")
    hco.set("xyz", "0 -0.045 0")
    hcg = ET.SubElement(hc, "geometry")
    hcy = ET.SubElement(hcg, "cylinder")
    hcy.set("length", _fmt(NOM_HIP_CYL_LEN * thigh_scale))
    hcy.set("radius", _fmt(NOM_HIP_CYL_R * thigh_scale))
    ixx, ixy, ixz, iyy, iyz, izz = _scale_I(
        0.00030528937, layout.hip_i_off[0], layout.hip_i_off[1],
        0.000590894859, layout.hip_i_off[2], 0.000396594572, 0.510299, m_hip,
    )
    _inertial(hip, layout.hip_com, m_hip, ixx, ixy, ixz, iyy, iyz, izz)

    _fixed(robot, f"{p}_hip_fixed", f"{p}_hip", f"{p}_thigh_shoulder", sh_xyz)
    sh = ET.SubElement(robot, "link")
    sh.set("name", f"{p}_thigh_shoulder")

    _revolute(robot, f"{p}_thigh_joint", f"{p}_hip", f"{p}_thigh", sh_xyz, "0 1 0", *THIGH_LIMIT)
    thigh = ET.SubElement(robot, "link")
    thigh.set("name", f"{p}_thigh")
    tv = ET.SubElement(thigh, "visual")
    tvo = ET.SubElement(tv, "origin")
    tvo.set("rpy", "0 0 0")
    tvo.set("xyz", "0 0 0")
    tg = ET.SubElement(tv, "geometry")
    tm = ET.SubElement(tg, "mesh")
    tm.set("filename", f"{mesh_p}/{layout.thigh_mesh}")
    tm.set("scale", "1 1 1")
    ET.SubElement(tv, "material").set("name", "silver")
    tc = ET.SubElement(thigh, "collision")
    tco = ET.SubElement(tc, "origin")
    tco.set("rpy", "0 1.57079632679 0")
    tco.set("xyz", f"0 0 {_fmt(-ul / 2)}")
    tcg = ET.SubElement(tc, "geometry")
    tbox = ET.SubElement(tcg, "box")
    tbox.set("size", f"{_fmt(ul)} {_fmt(0.0245 * thigh_scale)} {_fmt(0.034 * thigh_scale)}")
    oixy, oixz, oiyz = layout.thigh_i_off
    ixx, ixy, ixz, iyy, iyz, izz = _scale_I(
        0.005395867678, oixy, oixz, 0.005142451046, oiyz, 0.00102478732, 0.898919, m_th, ul / NOM_LEG_LEN,
    )
    _inertial(thigh, (-0.003468, layout.shoulder_sign * 0.018947, -0.032736 * thigh_scale), m_th, ixx, ixy, ixz, iyy, iyz, izz)

    _revolute(robot, f"{p}_calf_joint", f"{p}_thigh", f"{p}_calf", (0, 0, -ul), "0 1 0", *CALF_LIMIT)
    calf = ET.SubElement(robot, "link")
    calf.set("name", f"{p}_calf")
    cv = ET.SubElement(calf, "visual")
    cvo = ET.SubElement(cv, "origin")
    cvo.set("rpy", "0 0 0")
    cvo.set("xyz", "0 0 0")
    cg = ET.SubElement(cv, "geometry")
    cm = ET.SubElement(cg, "mesh")
    cm.set("filename", f"{mesh_p}/calf.stl")
    cm.set("scale", "1 1 1")
    ET.SubElement(cv, "material").set("name", "black")
    cc = ET.SubElement(calf, "collision")
    cco = ET.SubElement(cc, "origin")
    cco.set("rpy", "0 1.57079632679 0")
    cco.set("xyz", f"0 0 {_fmt(-ll / 2)}")
    ccg = ET.SubElement(cc, "geometry")
    cbox = ET.SubElement(ccg, "box")
    cbox.set("size", f"{_fmt(ll)} {_fmt(0.016 * (ll / NOM_LEG_LEN))} {_fmt(0.016 * (ll / NOM_LEG_LEN))}")
    lixx, lixy, lixz, liyy, liyz, lizz = _scale_I(
        0.003607648222, 1.494971e-06, -0.000132778525,
        0.003626771492, -2.8638535e-05, 3.5148003e-05, 0.158015, m_ca, ll / NOM_LEG_LEN,
    )
    _inertial(calf, (0.006286, 0.001307, -0.122269 * (ll / NOM_LEG_LEN)), m_ca, lixx, lixy, lixz, liyy, liyz, lizz)

    foot_r = max(lr * 1.5, 0.005)
    _fixed(robot, f"{p}_foot_fixed", f"{p}_calf", f"{p}_foot", (0, 0, -ll), dont_collapse=True)
    foot = ET.SubElement(robot, "link")
    foot.set("name", f"{p}_foot")
    fv = ET.SubElement(foot, "visual")
    fvo = ET.SubElement(fv, "origin")
    fvo.set("rpy", "0 0 0")
    fvo.set("xyz", "0 0 0")
    fvg = ET.SubElement(fv, "geometry")
    fvs = ET.SubElement(fvg, "sphere")
    fvs.set("radius", _fmt(0.01 * (ll / NOM_LEG_LEN)))
    ET.SubElement(fv, "material").set("name", "black")
    fc = ET.SubElement(foot, "collision")
    fco = ET.SubElement(fc, "origin")
    fco.set("rpy", "0 0 0")
    fco.set("xyz", "0 0 0")
    fcg = ET.SubElement(fc, "geometry")
    fcs = ET.SubElement(fcg, "sphere")
    fcs.set("radius", _fmt(foot_r))
    tix = 9.6e-06 * (m_foot / 0.06) * ((foot_r / NOM_TOE_COLL_RADIUS) ** 2)
    _inertial(foot, (0, 0, 0), m_foot, tix, 0, 0, tix, 0, tix)


def write_variant_urdf(path: Path, sample: QuadrupedVariantSample) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh_p = _mesh_prefix(path)
    bx, by, bz = sample.x_base * 2, sample.y_base * 2, sample.z_base * 2

    robot = ET.Element("robot")
    robot.set("name", "go1_description")

    _materials(robot)

    base = ET.SubElement(robot, "link")
    base.set("name", "base")
    bv = ET.SubElement(base, "visual")
    bvg = ET.SubElement(bv, "geometry")
    ET.SubElement(bvg, "box").set("size", "0.001 0.001 0.001")

    _fixed(robot, "floating_base", "base", "trunk", (0, 0, 0))

    trunk = ET.SubElement(robot, "link")
    trunk.set("name", "trunk")
    tv = ET.SubElement(trunk, "visual")
    tvg = ET.SubElement(tv, "geometry")
    tm = ET.SubElement(tvg, "mesh")
    tm.set("filename", f"{mesh_p}/trunk.stl")
    tm.set("scale", "1 1 1")
    ET.SubElement(tv, "material").set("name", "silver")
    tc = ET.SubElement(trunk, "collision")
    tbox = ET.SubElement(ET.SubElement(tc, "geometry"), "box")
    tbox.set("size", f"{_fmt(bx)} {_fmt(by)} {_fmt(bz)}")
    s_trunk = sample.base_mass / go1_base_mass
    ti = [x * s_trunk for x in (
        0.016130741919, 0.000593180607, 7.324662e-06,
        0.036507810812, 2.0969537e-05, 0.044693872053,
    )]
    _inertial(trunk, (0.011611, 0.004437, 0.000108), sample.base_mass, *ti)

    _fixed(robot, "imu_joint", "trunk", "imu_link", (-0.01592, -0.06659, -0.00617))
    imu = ET.SubElement(robot, "link")
    imu.set("name", "imu_link")
    m0 = float(sample.link_masses[0])
    ii = 1e-4 * (m0 / 0.001)
    _inertial(imu, (0, 0, 0), m0, ii, 0, 0, ii, 0, ii)
    iv = ET.SubElement(imu, "visual")
    ET.SubElement(ET.SubElement(iv, "geometry"), "box").set("size", "0.001 0.001 0.001")
    ET.SubElement(iv, "material").set("name", "red")

    ul = sample.upper_cylinder_length
    ur = sample.upper_cylinder_radius
    for i, lay in enumerate(_leg_layouts()):
        _add_leg(robot, sample, mesh_p, lay, i, float(ul[i]), float(ur[i]))

    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
