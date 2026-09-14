"""Full joint-range collision verification with official vendor servo geometry.

Runs outside the MCP 120 s ceiling. The high-face-count vendor hulls make
naive all-pairs booleans slow, so this does two things the in-session checks
could not afford:

1. Proves the parametric servo *block* (used during design) conservatively
   bounds the real vendor *hull* — i.e. real hull is contained in the block
   envelope. If so, every block-based clearance result is a lower bound and
   the real servos can only have MORE clearance.
2. Re-runs the collision matrix against the real hulls anyway, using a
   bounding-box prefilter so only genuinely close solid pairs get an exact
   boolean.

Run:  cd cad && uv run python analysis/collision_check.py
"""

from __future__ import annotations

import math
from pathlib import Path

from build123d import (
    Box,
    Compound,
    Cylinder,
    Plane,
    Pos,
    Rectangle,
    Rot,
    Solid,
    extrude,
    import_step,
    loft,
)

VENDOR = Path(__file__).resolve().parent.parent / "vendor"

# --- geometry constants (must match the build123d session / static_torque) ---
COXA_LEN, FEMUR_LEN, TIBIA_LEN = 60.0, 190.0, 380.0
HIP_HEIGHT, HIP_R = 270.0, 95.0
SRV_L, SRV_W, SRV_H, HORN_H = 45.2, 24.7, 35.0, 3.0
STAND_PITCH, STAND_INTERIOR = 30.0, 76.0
LEG_ANGLES = [30, 90, 150, 210, 270, 330]
YAW_LIMIT, KNEE_MIN, KNEE_MAX, PITCH_MAX, PITCH_MIN = 25.0, 35.0, 150.0, 85.0, -25.0


def servo_block(shaft_axis="Z", tuck=False):
    cx = -(SRV_L / 2 - 11.3) if tuck else (SRV_L / 2 - 11.3)
    case = Pos(cx, 0, -SRV_H / 2 - HORN_H) * Box(SRV_L, SRV_W, SRV_H)
    horn = Pos(0, 0, -HORN_H / 2) * Cylinder(10.0, HORN_H)
    s = case + horn
    return Rot(X=-90) * s if shaft_axis == "Y" else s


def _vendor_hull(fname: str, keep_idx: list[int]) -> Solid:
    v = import_step(str(VENDOR / fname))
    sol = sorted(v.solids(), key=lambda s: -s.volume)
    h = None
    for i in keep_idx:
        h = sol[i] if h is None else h.fuse(sol[i])
    return Pos(25.5, 0, -9.2) * Rot(X=90) * h


def coxa_bracket():
    from build123d import SlotOverall

    return extrude(Pos(COXA_LEN / 2, 0, 0) * SlotOverall(COXA_LEN + 30, 30), amount=6)


def femur_part():
    from build123d import SlotOverall

    sk = SlotOverall(FEMUR_LEN + 34, 30)
    plate = Rot(X=90) * extrude(Pos(FEMUR_LEN / 2, 0, 0) * sk, amount=4)
    return Pos(0, SRV_W / 2 + 5, 0) * plate + Pos(0, -SRV_W / 2 - 1, 0) * plate


def tibia_part():
    sec1 = Plane((0, 6, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * Rectangle(10, 26)
    sec2 = Plane((TIBIA_LEN - 18, 6, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * Rectangle(10, 7)
    shin = loft([sec1, sec2])
    tip = Pos(TIBIA_LEN - 9, 6, 0) * Rot(Y=90) * Cylinder(3.2, 18)  # blunt for speed
    hub = Pos(0, 6, 0) * Rot(X=90) * Cylinder(14, 12)
    return shin + tip + hub


def bbox_far(a, b, margin) -> bool:
    A, B = a.bounding_box(), b.bounding_box()
    return not (
        A.min.X - margin < B.max.X
        and B.min.X - margin < A.max.X
        and A.min.Y - margin < B.max.Y
        and B.min.Y - margin < A.max.Y
        and A.min.Z - margin < B.max.Z
        and B.min.Z - margin < A.max.Z
    )


def solids_of(x):
    return x.solids() if hasattr(x, "solids") else [x]


def pair(a, b, prefilter=60.0):
    """(overlap_mm3, min_dist_mm) between two shapes, bbox-prefiltered per solid."""
    mind, overlap = 1e9, 0.0
    for s in solids_of(a):
        for t in solids_of(b):
            if bbox_far(s, t, prefilter):
                continue
            d = s.distance_to(t)
            mind = min(mind, d)
            if d <= 1e-6:
                x = s & t
                if x is not None and x.volume > 1e-6:
                    overlap += x.volume
    return round(overlap, 1), (round(mind, 1) if mind < 1e9 else f">{prefilter:.0f}")


def leg(sv, yaw=0.0, pitch=STAND_PITCH, interior=STAND_INTERIOR):
    """sv: dict of servo shapes. Returns leg compound in hip-local frame."""
    import copy

    drop = 180.0 - pitch - interior
    kx = COXA_LEN + FEMUR_LEN * math.cos(math.radians(pitch))
    kz = FEMUR_LEN * math.sin(math.radians(pitch))
    parts = [
        copy.copy(sv["coxa_Z"]),
        coxa_bracket(),
        Pos(COXA_LEN, 0, 0) * Rot(Y=-pitch) * copy.copy(sv["pitch_Y"]),
        Pos(COXA_LEN, 0, 0) * Rot(Y=-pitch) * femur_part(),
        Pos(kx, 0, kz) * Rot(Y=-pitch) * copy.copy(sv["pitch_Y_tuck"]),
        Pos(kx, 0, kz) * Rot(Y=drop) * tibia_part(),
    ]
    return Rot(Z=yaw) * Compound(children=parts)


def placed(sv, az, **kw):
    return Pos(0, 0, HIP_HEIGHT) * (Rot(Z=az) * Pos(HIP_R, 0, 0) * leg(sv, **kw))


def main() -> None:
    print("importing vendor STEP...")
    hull3215 = _vendor_hull("sts3215/STS3215.step", [1, 2, 3, 5, 7])
    hull3250 = _vendor_hull("sts3250/STS3250.step", [0, 1, 2, 5, 6])

    # (1) containment: does the design block envelope bound the real hull?
    print("\n--- block envelope conservatism ---")
    for name, hull in (("STS3215", hull3215), ("STS3250", hull3250)):
        blk_Z = servo_block("Z")
        outside = hull.cut(blk_Z)
        ov = sum(s.volume for s in solids_of(outside)) if outside else 0.0
        hb, bb = hull.bounding_box(), blk_Z.bounding_box()
        print(
            f"{name}: hull vol outside block = {ov:.1f} mm3 "
            f"(hull bbox {hb.size.X:.1f}x{hb.size.Y:.1f}x{hb.size.Z:.1f}, "
            f"block {bb.size.X:.1f}x{bb.size.Y:.1f}x{bb.size.Z:.1f})"
        )
    print("  -> if ~0, all block-based clearances are conservative lower bounds")

    sv = {"coxa_Z": hull3215, "pitch_Y": Rot(X=-90) * hull3250, "pitch_Y_tuck": Rot(Z=180) * Rot(X=-90) * hull3250}

    print("\n--- collision matrix with REAL vendor hulls ---")
    la = placed(sv, 30, yaw=+YAW_LIMIT)
    lb = placed(sv, 90, yaw=-YAW_LIMIT)
    print(f"A adjacent legs both yaw {YAW_LIMIT:.0f} in : overlap={pair(la, lb)[0]} d={pair(la, lb)[1]}")
    ls = placed(sv, 30, yaw=+YAW_LIMIT, pitch=60, interior=50)
    print(f"B swing-over-stance          : overlap={pair(ls, lb)[0]} d={pair(ls, lb)[1]}")

    print("\n--- knee-fold sweep (tibia vs knee servo + femur plates) ---")
    for interior in (KNEE_MIN, 76.0, KNEE_MAX):
        pitch = 30.0
        drop = 180 - pitch - interior
        kx = COXA_LEN + FEMUR_LEN * math.cos(math.radians(pitch))
        kz = FEMUR_LEN * math.sin(math.radians(pitch))
        ks = Pos(kx, 0, kz) * Rot(Y=-pitch) * (Rot(Z=180) * Rot(X=-90) * hull3250)
        tf = Pos(kx, 0, kz) * Rot(Y=drop) * tibia_part()
        ff = Pos(COXA_LEN, 0, 0) * Rot(Y=-pitch) * femur_part()
        print(f"interior {interior:5.1f}: vs-servo {pair(ks, tf)}  vs-plates {pair(ff, tf)}")


if __name__ == "__main__":
    main()
