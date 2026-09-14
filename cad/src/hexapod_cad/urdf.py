"""URDF emission from the parametric model.

Emission, not conversion: joint frames, axes, and limits come from
``params.py`` directly — the same constants the solids are built from — so
nothing is re-derived lossily from geometry. Meshes are per-link STL
exported in link frame (mm, referenced with scale 0.001); collisions are
primitives written from parameters; inertials are exact ``massprops``
tensors with printed parts pinned to budget masses and servos pinned to
datasheet mass (vendor CAD volume is a lie — see vendor/*/PROVENANCE.md).

The output URDF is generated. Never edit it; edit params/links and rerun
``scripts/build_robot.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from build123d import export_stl

from hexapod_cad.links import link_body, link_coxa, link_femur, link_tibia
from hexapod_cad.massprops import MassProperties, compose, of_solid
from hexapod_cad.params import PARAMS, HexapodParams

MM = 1.0e-3


def pinned(shape, mass_g: float, label: str) -> MassProperties:
    """Geometry-shaped, mass-pinned properties (same trick as of_step's
    datasheet_mass): keep the distribution, set the magnitude."""
    raw = of_solid(shape, "pla_solid", label=label)  # density cancels below
    scale = (mass_g / 1000.0) / raw.mass
    return MassProperties(
        mass=mass_g / 1000.0, com=raw.com, inertia=raw.inertia * scale, volume=raw.volume, label=label
    )


def link_massprops(link: dict, label: str) -> MassProperties:
    return compose([pinned(s, m, f"{label}/{n}") for n, s, m in link["mass_items"]], label=label)


@dataclass
class LinkSpec:
    name: str
    mesh: str  # mesh filename (shared across legs)
    props: MassProperties
    collision_xml: str


def _inertial_xml(p: MassProperties) -> str:
    ixx, iyy, izz = p.inertia[0, 0], p.inertia[1, 1], p.inertia[2, 2]
    ixy, ixz, iyz = p.inertia[0, 1], p.inertia[0, 2], p.inertia[1, 2]
    c = p.com
    return f"""    <inertial>
      <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>
      <mass value="{p.mass:.6f}"/>
      <inertia ixx="{ixx:.3e}" ixy="{ixy:.3e}" ixz="{ixz:.3e}" iyy="{iyy:.3e}" iyz="{iyz:.3e}" izz="{izz:.3e}"/>
    </inertial>"""


def _link_xml(spec: LinkSpec, mesh_dir: str, material: str) -> str:
    return f"""  <link name="{spec.name}">
{_inertial_xml(spec.props)}
    <visual>
      <geometry><mesh filename="{mesh_dir}/{spec.mesh}" scale="0.001 0.001 0.001"/></geometry>
      <material name="{material}"/>
    </visual>
{spec.collision_xml}
  </link>"""


def _joint_xml(
    name: str,
    parent: str,
    child: str,
    xyz: tuple,
    rpy: tuple,
    axis: str,
    lo_deg: float,
    hi_deg: float,
    effort: float,
    vel: float,
) -> str:
    return f"""  <joint name="{name}" type="revolute">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>
    <axis xyz="{axis}"/>
    <limit lower="{math.radians(lo_deg):.6f}" upper="{math.radians(hi_deg):.6f}" effort="{effort}" velocity="{vel}"/>
    <dynamics damping="0.01" friction="0.0"/>
  </joint>"""


def _collisions(p: HexapodParams) -> dict[str, str]:
    """Primitive collision XML per link type, in link frame, metres.

    Cylinders become capsules at Isaac import (replace_cylinders_with_capsules).
    URDF cylinders are z-aligned; leg segments run along x -> rpy pitch 90 deg.
    """
    rot_x_to_z = f"0 {math.pi / 2:.6f} 0"
    body_r = p.body_across_flats / 2 / 0.866025 * MM
    servo = p.servo_pitch
    # coxa: one box bounding bracket + hip servo case (z -39 .. +6 around axis)
    cox_l = (p.coxa_len + servo.case_l / 2 - servo.shaft_from_rear + servo.case_l / 2) * MM
    return {
        "body": f"""    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="{body_r:.4f}" length="{(p.body_h + 20) * MM:.4f}"/></geometry>
    </collision>""",
        "coxa": f"""    <collision>
      <origin xyz="{(p.coxa_len + 12) / 2 * MM:.4f} 0 {-16.5 * MM:.4f}" rpy="0 0 0"/>
      <geometry><box size="{cox_l:.4f} {(servo.case_w + 2) * MM:.4f} {45 * MM:.4f}"/></geometry>
    </collision>""",
        "femur": f"""    <collision>
      <origin xyz="{p.femur_len / 2 * MM:.4f} 0 0" rpy="{rot_x_to_z}"/>
      <geometry><cylinder radius="{(p.plate_gap_inner + p.plate_t) * MM:.4f}"
          length="{(p.femur_len + 34) * MM:.4f}"/></geometry>
    </collision>""",
        "tibia": f"""    <collision>
      <origin xyz="{(p.tibia_len - 20) / 2 * MM:.4f} {6 * MM:.4f} 0" rpy="{rot_x_to_z}"/>
      <geometry><cylinder radius="{9 * MM:.4f}" length="{(p.tibia_len - 20) * MM:.4f}"/></geometry>
    </collision>
    <collision>
      <origin xyz="{(p.tibia_len - 3) * MM:.4f} {6 * MM:.4f} 0" rpy="0 0 0"/>
      <geometry><sphere radius="{4 * MM:.4f}"/></geometry>
    </collision>""",
    }


def build(out_urdf: Path, mesh_out_dir: Path, mesh_ref_dir: str, p: HexapodParams = PARAMS) -> dict:
    """Export per-link STLs and emit the URDF. Returns a build report dict."""
    mesh_out_dir.mkdir(parents=True, exist_ok=True)

    builders = {"body": link_body, "coxa": link_coxa, "femur": link_femur, "tibia": link_tibia}
    links, report = {}, {}
    for kind, fn in builders.items():
        data = fn(p)
        mesh = f"{kind}.stl"
        export_stl(data["visual"], str(mesh_out_dir / mesh))
        props = link_massprops(data, kind)
        assert props.is_physically_valid(), f"{kind}: invalid inertia tensor"
        links[kind] = LinkSpec(kind, mesh, props, _collisions(p)[kind])
        report[kind] = {
            "mass_g": props.mass * 1e3,
            "com_mm": [round(v * 1e3, 1) for v in props.com],
            "mesh_kb": round((mesh_out_dir / mesh).stat().st_size / 1024, 1),
        }

    parts = [
        '<?xml version="1.0"?>',
        '<robot name="spidertron_hexapod">',
        '  <material name="pod_grey"><color rgba="0.55 0.57 0.60 1"/></material>',
        '  <material name="leg_orange"><color rgba="0.85 0.45 0.10 1"/></material>',
        '  <material name="leg_dark"><color rgba="0.20 0.20 0.22 1"/></material>',
    ]

    parts.append(
        _link_xml(
            LinkSpec("base_link", "body.stl", links["body"].props, links["body"].collision_xml),
            mesh_ref_dir,
            "pod_grey",
        )
    )

    fe, kv = p.servo_pitch.effort_limit_nm, p.servo_pitch.velocity_limit_rad_s
    ce, cv = p.servo_yaw.effort_limit_nm, p.servo_yaw.velocity_limit_rad_s
    for leg, az_deg in p.legs.items():
        az = math.radians(az_deg)
        for kind, mat in (("coxa", "leg_dark"), ("femur", "leg_orange"), ("tibia", "leg_dark")):
            spec = links[kind]
            parts.append(
                _link_xml(LinkSpec(f"{leg}_{kind}", spec.mesh, spec.props, spec.collision_xml), mesh_ref_dir, mat)
            )
        parts.append(
            _joint_xml(
                f"{leg}_coxa_joint",
                "base_link",
                f"{leg}_coxa",
                (p.hip_r * MM * math.cos(az), p.hip_r * MM * math.sin(az), 0.0),
                (0.0, 0.0, az),
                "0 0 1",
                *p.coxa_range_deg,
                ce,
                cv,
            )
        )
        parts.append(
            _joint_xml(
                f"{leg}_femur_joint",
                f"{leg}_coxa",
                f"{leg}_femur",
                (p.coxa_len * MM, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                "0 1 0",
                *p.femur_joint_range_deg,
                fe,
                kv,
            )
        )
        parts.append(
            _joint_xml(
                f"{leg}_tibia_joint",
                f"{leg}_femur",
                f"{leg}_tibia",
                (p.femur_len * MM, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                "0 1 0",
                *p.knee_range_deg,
                fe,
                kv,
            )
        )
    parts.append("</robot>")

    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_urdf.write_text("\n".join(parts), encoding="utf-8", newline="\n")

    total = links["body"].props.mass + 6 * sum(links[k].props.mass for k in ("coxa", "femur", "tibia"))
    report["totals"] = {
        "robot_mass_g": round(total * 1e3, 1),
        "links": 1 + 18,
        "joints": 18,
        "default_joint_pose_rad": {k: round(v, 4) for k, v in p.default_joint_pose_rad().items()},
        "foot_z_standing_mm": round(p.foot_z_standing(), 2),
    }
    return report
