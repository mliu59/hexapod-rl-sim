"""Per-link solid builders, each in its own URDF link frame, zero pose.

Link frames (URDF convention: link frame = parent joint frame):
  base_link : body centre, z=0 at the hip-axis plane
  *_coxa    : origin at the yaw axis, +x radially outward, +z up
  *_femur   : origin at the femur pitch axis, +x toward the knee (zero pose
              = horizontal); joint rotation about +y, negative lifts
  *_tibia   : origin at the knee axis, +x toward the foot (zero pose =
              aligned with the femur extension)

All six legs share identical geometry (radial symmetry, no mirroring), so
one STL per link type serves all legs.

The hip pitch servo CASE lives in the coxa link (bolted to the bracket; the
horn drives the femur) and the knee servo case lives in the femur link.
This is the physically correct stator-side assignment — the interactive
session's mockup drew the hip case rotating with the femur, corrected here.

Geometry is the collision-verified session design (DESIGN_LOG iterations
1-3): vertical twin plates straddling the hip servo, knee case tucked back
along the femur, single-side tibia arm riding the horn face, low-poly servo
envelopes measured from the official Feetech STEP.
"""

from __future__ import annotations

from build123d import (
    Axis,
    Box,
    Compound,
    Cylinder,
    Part,
    Plane,
    Pos,
    Rectangle,
    RegularPolygon,
    Rot,
    SlotOverall,
    Sphere,
    chamfer,
    extrude,
    loft,
)

from hexapod_cad.params import PARAMS, HexapodParams, ServoSpec


def servo_envelope(spec: ServoSpec, shaft_axis: str = "Z", tuck: bool = False) -> Part:
    """Low-poly servo (10 faces): exact vendor outer envelope.

    Shaft +Z through origin, horn interface plane at z=0, case below.
    Verified conservative superset of the official hull (sub-mm slivers only).
    tuck mirrors the case to -x (knee mount, case back along the femur).
    """
    cx = spec.case_l / 2 - spec.shaft_from_rear
    case = Pos(cx, 0, -(spec.case_h / 2 + spec.horn_h)) * Box(spec.case_l, spec.case_w, spec.case_h)
    horn = Pos(0, 0, -spec.horn_h / 2) * Cylinder(spec.horn_r, spec.horn_h)
    s = case + horn
    if tuck:
        s = Rot(Z=180) * s
    if shaft_axis == "Y":
        s = Rot(X=-90) * s
    return s


# ---------------------------------------------------------------------------
# printed parts (one function per printed component, in link frame)
# ---------------------------------------------------------------------------


def bracket(p: HexapodParams = PARAMS) -> Part:
    """Coxa bracket: slab from the yaw horn out to the femur axis, z in [0, t]."""
    sk = SlotOverall(p.coxa_len + p.bracket_w, p.bracket_w)
    return extrude(Pos(p.coxa_len / 2, 0, 0) * sk, amount=p.bracket_t)


def femur_plates(p: HexapodParams = PARAMS) -> Part:
    """Vertical twin plates along +x, straddling the hip servo case.

    The raw plate solid spans y in [-t, 0]; inner faces land at +/-gap_inner.
    """
    sk = SlotOverall(p.femur_len + 34, p.plate_h)
    plate = Rot(X=90) * extrude(Pos(p.femur_len / 2, 0, 0) * sk, amount=p.plate_t)
    return Pos(0, p.plate_gap_inner + p.plate_t, 0) * plate + Pos(0, -p.plate_gap_inner, 0) * plate


def tibia_arm(p: HexapodParams = PARAMS) -> Part:
    """Single-side tapered arm on the horn face: y in [1, 11], pointed foot."""
    y_mid = 1.0 + p.tibia_arm_w / 2
    sec1 = Plane((0, y_mid, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * Rectangle(p.tibia_arm_w, p.tibia_arm_root_h)
    sec2 = Plane((p.tibia_len - 18, y_mid, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * Rectangle(
        p.tibia_arm_w, p.tibia_arm_tip_h
    )
    shin = loft([sec1, sec2])
    tip = Pos(p.tibia_len - 9, y_mid, 0) * Rot(Y=90) * Cylinder(3.2, 18)
    hub = Pos(0, p.tibia_hub_t / 2, 0) * Rot(X=90) * Cylinder(p.tibia_hub_r, p.tibia_hub_t)
    return shin + tip + hub


def body_shell(p: HexapodParams = PARAMS) -> Part:
    """Hex pod with chamfers, dome, and hip servo bays. Centred, hip plane z=0."""
    circum_r = p.body_across_flats / 2 / 0.866025
    pod = extrude(RegularPolygon(circum_r, 6, rotation=30), amount=p.body_h / 2, both=True)
    pod = chamfer(pod.edges().group_by(Axis.Z)[0], length=18)
    pod = chamfer(pod.edges().group_by(Axis.Z)[-1], length=14)
    dome = Pos(0, 0, p.body_h / 2) * Cylinder(p.dome_r, 20) + Pos(0, 0, p.body_h / 2 + 20) * Sphere(
        p.dome_r, arc_size1=0, arc_size2=90
    )
    body = pod + dome
    for az in p.legs.values():
        body -= Rot(Z=az) * Pos(p.hip_r, 0, -14) * Cylinder(p.hip_bay_r, 60)
    return body


def electronics_lump(p: HexapodParams = PARAMS) -> Part:
    """Battery + controller + wiring as one box inside the pod (mass only,
    excluded from the visual mesh)."""
    return Pos(0, 0, -5) * Box(90, 60, 45)


# ---------------------------------------------------------------------------
# link assemblies: (visual solids, mass items) per link
# ---------------------------------------------------------------------------


def link_body(p: HexapodParams = PARAMS) -> dict:
    """base_link: shell + the six yaw servo cases bolted into the bays."""
    yaw_cases = [Rot(Z=az) * Pos(p.hip_r, 0, 0) * servo_envelope(p.servo_yaw) for az in p.legs.values()]
    return {
        "visual": Compound(children=[body_shell(p)] + yaw_cases),
        "mass_items": (
            [("shell", body_shell(p), p.mass_body_shell_g)]
            + [("electronics", electronics_lump(p), p.mass_electronics_g)]
            + [
                (
                    f"yaw_servo_{az:.0f}",
                    Rot(Z=az) * Pos(p.hip_r, 0, 0) * servo_envelope(p.servo_yaw),
                    p.servo_yaw.mass_g,
                )
                for az in p.legs.values()
            ]
        ),
    }


def link_coxa(p: HexapodParams = PARAMS) -> dict:
    """Coxa: bracket + hip pitch servo case (stator side), shaft along y at x=coxa_len."""
    hip_case = Pos(p.coxa_len, 0, 0) * servo_envelope(p.servo_pitch, shaft_axis="Y")
    return {
        "visual": Compound(children=[bracket(p), hip_case]),
        "mass_items": [
            ("bracket", bracket(p), p.mass_bracket_g),
            ("hip_servo", hip_case, p.servo_pitch.mass_g),
        ],
    }


def link_femur(p: HexapodParams = PARAMS) -> dict:
    """Femur: twin plates + knee servo case tucked back along the femur."""
    knee_case = Pos(p.femur_len, 0, 0) * servo_envelope(p.servo_pitch, shaft_axis="Y", tuck=True)
    return {
        "visual": Compound(children=[femur_plates(p), knee_case]),
        "mass_items": [
            ("plates", femur_plates(p), p.mass_femur_g),
            ("knee_servo", knee_case, p.servo_pitch.mass_g),
        ],
    }


def link_tibia(p: HexapodParams = PARAMS) -> dict:
    return {
        "visual": tibia_arm(p),
        "mass_items": [("arm", tibia_arm(p), p.mass_tibia_g)],
    }
