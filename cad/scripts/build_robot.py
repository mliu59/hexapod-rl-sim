"""Build the spidertron hexapod URDF + per-link STL meshes from the
parametric source. Rerunnable; this is THE way robot iterations become
URDFs — edit hexapod_cad/params.py (or links.py) and rerun.

Usage:
    cd cad && uv run python scripts/build_robot.py
        -> ../assets/urdf/spidertron.urdf, ../assets/meshes/spidertron/*.stl

Validation is part of the build: yourdfpy parse + kinematic-tree check, FK
standing-pose foot height, mass totals vs the torque-analysis budget. A
build that fails validation exits nonzero and writes nothing misleading.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def validate(urdf_path: Path, report: dict) -> list[str]:
    errors: list[str] = []

    import yourdfpy

    model = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_scene_graph=True)
    if not model.validate():
        errors.append("yourdfpy.validate() failed")
    nj = len([j for j in model.robot.joints if j.type == "revolute"])
    if nj != 18:
        errors.append(f"expected 18 revolute joints, got {nj}")
    if len(model.robot.links) != 19:
        errors.append(f"expected 19 links, got {len(model.robot.links)}")

    # FK at the standing pose: every foot tip must land at z ~= -hip_height
    from hexapod_cad.params import PARAMS

    pose = PARAMS.default_joint_pose_rad()
    cfg = {}
    for leg in PARAMS.legs:
        cfg[f"{leg}_coxa_joint"] = pose["coxa"]
        cfg[f"{leg}_femur_joint"] = pose["femur"]
        cfg[f"{leg}_tibia_joint"] = pose["tibia"]
    model.update_cfg(cfg)
    tip_local = [(PARAMS.tibia_len + 1.0) / 1000.0, 6.0 / 1000.0, 0.0, 1.0]
    import numpy as np

    want_z = -PARAMS.hip_height / 1000.0
    for leg in PARAMS.legs:
        T = model.get_transform(f"{leg}_tibia", "base_link")
        z = float((T @ np.array(tip_local))[2])
        if abs(z - want_z) > 0.003:
            errors.append(f"{leg} foot z {z:.4f} m, want {want_z:.4f} m")

    # mass total against the torque-analysis budget
    total = report["totals"]["robot_mass_g"]
    if not (2800.0 <= total <= 3000.0):
        errors.append(f"robot mass {total:.0f} g outside 2800-3000 g budget window")
    return errors


def main() -> int:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", type=Path, default=REPO / "assets" / "urdf" / "spidertron.urdf")
    ap.add_argument("--mesh-dir", type=Path, default=REPO / "assets" / "meshes" / "spidertron")
    args = ap.parse_args()

    from hexapod_cad.urdf import build

    mesh_ref = "../meshes/spidertron"  # relative to the URDF file location
    print(f"building -> {args.urdf}")
    report = build(args.urdf, args.mesh_dir, mesh_ref)

    for kind in ("body", "coxa", "femur", "tibia"):
        r = report[kind]
        print(f"  {kind:6s}: {r['mass_g']:7.1f} g  com {r['com_mm']} mm  mesh {r['mesh_kb']} KB")
    t = report["totals"]
    print(f"  total : {t['robot_mass_g']:.1f} g, {t['links']} links / {t['joints']} joints")
    print(f"  standing pose (rad): {t['default_joint_pose_rad']}  foot z: {t['foot_z_standing_mm']} mm below hip")

    errors = validate(args.urdf, report)
    if errors:
        print("\nVALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        return 1
    print("\nvalidation: URDF parses, tree correct, 6 feet on the ground, mass in budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
