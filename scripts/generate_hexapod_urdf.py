"""Parametric placeholder hexapod URDF generator.

Stand-in for the real hexapod model (built in a separate project) so the
Isaac Lab pipeline work is not blocked on it. Everything downstream —
URDF->USD import, actuator config, env config, rewards — is identical for
this model and the final one.

Layout: rectangular body, 6 legs (3 per side), 3 DOF each = 18 DOF.
  coxa  : yaw about z   (leg swing, "hip abduction" analog)
  femur : pitch about y (leg lift)
  tibia : pitch about y (knee)

All collision geometry is primitives (box / cylinder / sphere). Inertia
tensors are computed analytically from the primitive geometry and the link
mass (uniform density) — never guessed.

Scale/mass target: hobby hexapod in the PhantomX class (~3 kg total),
consistent with Dynamixel XM430-class actuators (see ACTUATOR NOTES below).

Usage:
    python scripts/generate_hexapod_urdf.py [-o assets/urdf/hexapod.urdf]

ACTUATOR NOTES (Dynamixel XM430-W350, 12 V — used later in the Isaac Lab
ImplicitActuatorCfg, recorded here so URDF <limit> tags stay consistent):
  stall torque   : 4.1 N*m   -> effort limit 3.0 N*m (sustained, derated)
  no-load speed  : 46 rpm    -> velocity limit 4.8 rad/s
  suggested PD   : stiffness Kp = 25.0 N*m/rad, damping Kd = 0.5 N*m*s/rad
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------

@dataclass
class HexapodParams:
    # body (box, meters)
    body_length: float = 0.30
    body_width: float = 0.22
    body_height: float = 0.06
    body_mass: float = 1.20  # kg — frame + electronics + battery

    # leg segments (cylinders along local +x, meters)
    coxa_length: float = 0.055
    coxa_radius: float = 0.018
    coxa_mass: float = 0.08  # kg — bracket + share of servo mass

    femur_length: float = 0.10
    femur_radius: float = 0.015
    femur_mass: float = 0.12

    tibia_length: float = 0.14
    tibia_radius: float = 0.012
    tibia_mass: float = 0.10

    foot_radius: float = 0.012

    # hip placement: (x, y) of each left-side hip on the body, and the yaw
    # of the leg mount frame (deg from body +x). Right side is mirrored.
    hip_x: tuple = (0.12, 0.0, -0.12)
    hip_y: float = 0.09
    mount_yaw_deg: tuple = (50.0, 90.0, 130.0)  # front, middle, rear

    # joint limits (rad) and actuator limits (XM430-class, see header)
    coxa_limit: tuple = (-0.9, 0.9)
    femur_limit: tuple = (-1.6, 1.6)
    tibia_limit: tuple = (-0.2, 2.7)
    effort_limit: float = 3.0     # N*m
    velocity_limit: float = 4.8   # rad/s
    joint_damping: float = 0.01   # passive, in URDF; PD gains live in Isaac Lab cfg
    joint_friction: float = 0.0


# Default standing pose (rad) — not part of URDF, consumed by the Isaac Lab
# ArticulationCfg init_state. Recorded here as the single source of truth.
# femur -0.35 lifts the femur slightly above horizontal; tibia +1.75 folds the
# knee down so the foot sits ~0.10 m below the hip.
DEFAULT_JOINT_POSE = {"coxa": 0.0, "femur": -0.35, "tibia": 1.75}


# ----------------------------------------------------------------------------
# Inertia helpers (uniform density, about link COM, principal axes)
# ----------------------------------------------------------------------------

def box_inertia(m: float, lx: float, ly: float, lz: float) -> tuple:
    ixx = m / 12.0 * (ly**2 + lz**2)
    iyy = m / 12.0 * (lx**2 + lz**2)
    izz = m / 12.0 * (lx**2 + ly**2)
    return ixx, iyy, izz


def cylinder_x_inertia(m: float, r: float, length: float) -> tuple:
    """Cylinder whose axis is the local +x axis."""
    ixx = 0.5 * m * r**2
    iyy = izz = m / 12.0 * (3.0 * r**2 + length**2)
    return ixx, iyy, izz


# ----------------------------------------------------------------------------
# URDF emission
# ----------------------------------------------------------------------------

def _inertial(m: float, com_xyz: str, inertia: tuple) -> str:
    ixx, iyy, izz = inertia
    return f"""    <inertial>
      <origin xyz="{com_xyz}" rpy="0 0 0"/>
      <mass value="{m}"/>
      <inertia ixx="{ixx:.6e}" ixy="0" ixz="0" iyy="{iyy:.6e}" iyz="0" izz="{izz:.6e}"/>
    </inertial>"""


def _cylinder_link(name: str, m: float, r: float, length: float,
                   material: str, foot_radius: float | None = None) -> str:
    """Link with a cylinder along +x from origin to (length, 0, 0).

    URDF cylinders are z-aligned, so the geometry origin pitches z onto x
    (rpy pitch = pi/2) and sits at the segment midpoint.
    """
    geo_origin = f'<origin xyz="{length / 2.0:.4f} 0 0" rpy="0 {math.pi / 2.0:.8f} 0"/>'
    cyl = f'<cylinder radius="{r}" length="{length}"/>'
    foot = ""
    if foot_radius is not None:
        foot = f"""
    <collision>
      <origin xyz="{length:.4f} 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="{foot_radius}"/></geometry>
    </collision>
    <visual>
      <origin xyz="{length:.4f} 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="{foot_radius}"/></geometry>
      <material name="{material}"/>
    </visual>"""
    return f"""  <link name="{name}">
{_inertial(m, f"{length / 2.0:.4f} 0 0", cylinder_x_inertia(m, r, length))}
    <visual>
      {geo_origin}
      <geometry>{cyl}</geometry>
      <material name="{material}"/>
    </visual>
    <collision>
      {geo_origin}
      <geometry>{cyl}</geometry>
    </collision>{foot}
  </link>"""


def _revolute_joint(name: str, parent: str, child: str, xyz: str, rpy: str,
                    axis: str, limits: tuple, p: HexapodParams) -> str:
    lo, hi = limits
    return f"""  <joint name="{name}" type="revolute">
    <origin xyz="{xyz}" rpy="{rpy}"/>
    <parent link="{parent}"/>
    <child link="{child}"/>
    <axis xyz="{axis}"/>
    <limit lower="{lo}" upper="{hi}" effort="{p.effort_limit}" velocity="{p.velocity_limit}"/>
    <dynamics damping="{p.joint_damping}" friction="{p.joint_friction}"/>
  </joint>"""


def generate(p: HexapodParams) -> str:
    parts: list[str] = []
    parts.append('<?xml version="1.0"?>')
    parts.append("<!-- AUTO-GENERATED by scripts/generate_hexapod_urdf.py - do not hand-edit -->")
    parts.append('<robot name="hexapod">')
    parts.append("""  <material name="body_grey"><color rgba="0.35 0.35 0.38 1"/></material>
  <material name="leg_orange"><color rgba="0.85 0.45 0.10 1"/></material>
  <material name="leg_dark"><color rgba="0.20 0.20 0.22 1"/></material>""")

    # body
    bi = box_inertia(p.body_mass, p.body_length, p.body_width, p.body_height)
    parts.append(f"""  <link name="base_link">
{_inertial(p.body_mass, "0 0 0", bi)}
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{p.body_length} {p.body_width} {p.body_height}"/></geometry>
      <material name="body_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{p.body_length} {p.body_width} {p.body_height}"/></geometry>
    </collision>
  </link>""")

    # legs: L/R x front/middle/rear
    positions = ["F", "M", "R"]
    for side, sign in (("L", 1.0), ("R", -1.0)):
        for i, pos in enumerate(positions):
            leg = f"{side}{pos}"
            hip_x = p.hip_x[i]
            hip_y = sign * p.hip_y
            yaw = sign * math.radians(p.mount_yaw_deg[i])

            parts.append(_revolute_joint(
                f"{leg}_coxa_joint", "base_link", f"{leg}_coxa",
                f"{hip_x} {hip_y} 0", f"0 0 {yaw:.8f}", "0 0 1",
                p.coxa_limit, p))
            parts.append(_cylinder_link(f"{leg}_coxa", p.coxa_mass,
                                        p.coxa_radius, p.coxa_length, "leg_dark"))

            parts.append(_revolute_joint(
                f"{leg}_femur_joint", f"{leg}_coxa", f"{leg}_femur",
                f"{p.coxa_length} 0 0", "0 0 0", "0 1 0",
                p.femur_limit, p))
            parts.append(_cylinder_link(f"{leg}_femur", p.femur_mass,
                                        p.femur_radius, p.femur_length, "leg_orange"))

            parts.append(_revolute_joint(
                f"{leg}_tibia_joint", f"{leg}_femur", f"{leg}_tibia",
                f"{p.femur_length} 0 0", "0 0 0", "0 1 0",
                p.tibia_limit, p))
            parts.append(_cylinder_link(f"{leg}_tibia", p.tibia_mass,
                                        p.tibia_radius, p.tibia_length, "leg_dark",
                                        foot_radius=p.foot_radius))

    parts.append("</robot>")
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", type=Path,
                    default=Path(__file__).resolve().parent.parent / "assets" / "urdf" / "hexapod.urdf")
    args = ap.parse_args()

    p = HexapodParams()
    urdf = generate(p)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(urdf, encoding="utf-8", newline="\n")

    total_mass = p.body_mass + 6 * (p.coxa_mass + p.femur_mass + p.tibia_mass)
    print(f"wrote {args.output}")
    print(f"links: {1 + 18}, joints: 18, total mass: {total_mass:.2f} kg")
    print(f"default joint pose (rad): {DEFAULT_JOINT_POSE}")


if __name__ == "__main__":
    main()
