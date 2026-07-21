"""Render the hexapod URDF with matplotlib — no Isaac Sim / RTX involved.

Parses assets/urdf/hexapod.urdf directly, applies the default standing pose
(DEFAULT_JOINT_POSE from generate_hexapod_urdf.py), and draws the primitive
geometry (box body, cylinder legs, sphere feet) in 3D. Useful while the
Omniverse RTX renderer is unavailable (driver incompatibility) and generally
as a fast geometry check after editing the generator.

Usage:
    python scripts/render_urdf.py [-o docs/hexapod_render.png] [--pose zero|stand]
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
STAND_POSE = {"coxa": 0.0, "femur": -0.35, "tibia": 1.75}  # keep in sync with generator


def rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def tf(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = xyz
    return m


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )


def parse_floats(s: str | None, default: str = "0 0 0") -> np.ndarray:
    return np.array([float(v) for v in (s or default).split()])


def joint_angle(joint_name: str, pose: dict[str, float]) -> float:
    for key, val in pose.items():
        if key in joint_name:
            return val
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", type=Path, default=REPO / "assets" / "urdf" / "hexapod.urdf")
    ap.add_argument("-o", "--output", type=Path, default=REPO / "docs" / "hexapod_render.png")
    ap.add_argument("--pose", choices=["stand", "zero"], default="stand")
    args = ap.parse_args()

    pose = STAND_POSE if args.pose == "stand" else {}
    root = ET.parse(args.input).getroot()

    joints = {}  # child link -> (parent link, origin_tf, axis)
    for j in root.findall("joint"):
        o = j.find("origin")
        origin = tf(parse_floats(o.get("xyz")), rpy_to_mat(*parse_floats(o.get("rpy"))))
        joints[j.find("child").get("link")] = (
            j.find("parent").get("link"),
            origin,
            parse_floats(j.find("axis").get("xyz"), "0 0 1"),
            j.get("name"),
        )

    # world transform of each link (base at standing height for ground context)
    base_h = 0.111 if args.pose == "stand" else 0.2
    world = {"base_link": tf(np.array([0, 0, base_h]), np.eye(3))}

    def link_tf(name: str) -> np.ndarray:
        if name not in world:
            parent, origin, axis, jname = joints[name]
            world[name] = link_tf(parent) @ origin @ tf(np.zeros(3), axis_angle(axis, joint_angle(jname, pose)))
        return world[name]

    fig = plt.figure(figsize=(14, 6))
    views = [("isometric", 25, -55), ("front", 5, -90), ("top", 88, -90)]
    for idx, (title, elev, azim) in enumerate(views, 1):
        axp = fig.add_subplot(1, 3, idx, projection="3d")
        for link in root.findall("link"):
            name = link.get("name")
            T = link_tf(name)
            for vis in link.findall("visual"):
                o = vis.find("origin")
                Tg = T @ tf(parse_floats(o.get("xyz")), rpy_to_mat(*parse_floats(o.get("rpy"))))
                geo = vis.find("geometry")
                box, cyl, sph = geo.find("box"), geo.find("cylinder"), geo.find("sphere")
                if box is not None:
                    sx, sy, sz = parse_floats(box.get("size")) / 2
                    corners = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
                    pts = (Tg[:3, :3] @ corners.T).T + Tg[:3, 3]
                    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
                                 (0, 4), (1, 5), (2, 6), (3, 7)]:
                        axp.plot(*zip(pts[a], pts[b]), color="dimgray", lw=1.5)
                elif cyl is not None:
                    r, length = float(cyl.get("radius")), float(cyl.get("length"))
                    a = Tg @ np.array([0, 0, -length / 2, 1])
                    b = Tg @ np.array([0, 0, length / 2, 1])
                    axp.plot(*zip(a[:3], b[:3]), color="darkorange", lw=1 + 400 * r, solid_capstyle="round")
                elif sph is not None:
                    axp.scatter(*Tg[:3, 3], s=(400 * float(sph.get("radius"))) ** 2, color="black")
        # ground
        g = np.linspace(-0.35, 0.35, 2)
        gx, gy = np.meshgrid(g, g)
        axp.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.15, color="green")
        axp.set_title(f"{title} ({args.pose} pose)")
        axp.set_xlim(-0.35, 0.35), axp.set_ylim(-0.35, 0.35), axp.set_zlim(0, 0.45)
        axp.set_box_aspect((1, 1, 0.64))
        axp.view_init(elev=elev, azim=azim)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
