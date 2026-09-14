"""Static torque sizing for the spidertron-style hexapod.

Answers one question: what stall torque must each joint's servo have, given
the leg geometry the 20 cm obstacle requirement forces on us and the mass the
robot will plausibly be. Statics only — no dynamics, no impacts. The safety
factors at the end are what absorb that simplification.

Method
------
Vertical ground-reaction force F at the foot; each pitch joint's torque is
F x (horizontal distance from that joint's axis to the foot). We sweep stance
poses (foot radius x body height) rather than trusting one hand-picked pose,
and take the worst reachable pose as the sizing case.

Load cases:
  nominal   - tripod gait, weight on 3 legs             -> F = W/3
  worst     - rough terrain / gait transition, 2 legs   -> F = W/2
  swing     - leg fully extended horizontally, lifting
              only its own mass (worst femur arm, no
              ground reaction)

Coxa (yaw, vertical axis) sees no gravity moment; it is sized by propulsion/
friction shear at the foot, taken as 0.5*F acting at the full horizontal
reach.

Run:  cd cad && uv run python analysis/static_torque.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81

# ---------------------------------------------------------------------------
# Geometry (mm) — must match hexapod_cad's part parameters when those exist
# ---------------------------------------------------------------------------

COXA_LEN = 60.0  # yaw axis -> femur pitch axis
FEMUR_LEN = 190.0  # femur pitch axis -> knee axis
TIBIA_LEN = 380.0  # knee axis -> foot tip
BODY_CLEAR = 240.0  # standing clearance under the body pod (> 200 obstacle)
HIP_HEIGHT = 270.0  # coxa/femur axis height when standing

# Operating envelope. An unconstrained sweep found the sizing pose to be a
# fully-splayed crouch (reach 340 mm at hip 200 mm) that pushes the femur to
# 49 kg*cm — a pose with no functional purpose. We exclude it BY DESIGN and
# the exclusion must be enforced downstream: URDF joint limits + gait/stance
# envelope in the RL env. These are load-bearing numbers, not suggestions.
REACH_MAX = 280.0  # max horizontal femur-axis -> foot in stance
REACH_MIN = 140.0
HIP_MIN = 210.0  # deepest crouch (body still clears 180 mm terrain)

# ---------------------------------------------------------------------------
# Mass budget (g)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Servo:
    name: str
    stall_kgcm: float  # stall torque at listed voltage
    rated_kgcm: float  # sustained ("rated") torque
    mass_g: float
    speed_s_per_60deg: float
    price_usd: float
    volts: float


# Candidates. Stall figures from datasheets/servodatabase; rated figures are
# the manufacturers' sustained numbers where published, else ~25% of stall
# (the usual serial-bus servo derating).
STS3215_12V = Servo("Feetech STS3215 (12 V)", 30.0, 8.0, 60.0, 0.146, 16.0, 12.0)
STS3250_12V = Servo("Feetech STS3250 (12 V)", 50.0, 13.0, 74.5, 0.133, 40.0, 12.0)
XL430 = Servo("Dynamixel XL430-W250", 15.3, 4.0, 57.2, 0.164, 50.0, 12.0)
XM430 = Servo("Dynamixel XM430-W350", 41.8, 12.0, 82.0, 0.113, 270.0, 12.0)

CANDIDATES = [STS3215_12V, STS3250_12V, XL430, XM430]

# Proposed assignment (challenged by the results table below)
COXA_SERVO = STS3215_12V
FEMUR_SERVO = STS3250_12V
KNEE_SERVO = STS3250_12V

PLA_COXA_G = 30.0  # printed bracket around the two hip servos
PLA_FEMUR_G = 60.0  # twin-plate femur, 190 mm
PLA_TIBIA_G = 85.0  # tapered tibia + TPU foot tip, 380 mm
BODY_SHELL_G = 220.0  # hex pod shell + lids + servo cage
ELECTRONICS_G = 380.0  # 3S battery ~180, controller + bus driver ~100, wiring ~100


def leg_mass_g() -> float:
    return COXA_SERVO.mass_g + FEMUR_SERVO.mass_g + KNEE_SERVO.mass_g + PLA_COXA_G + PLA_FEMUR_G + PLA_TIBIA_G


def robot_mass_g() -> float:
    return BODY_SHELL_G + ELECTRONICS_G + 6 * leg_mass_g()


# ---------------------------------------------------------------------------
# Kinematics: stance poses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """A stance pose of one leg, all lengths mm, angles rad."""

    femur_pitch: float  # + above horizontal
    knee_drop: float  # tibia angle below horizontal
    foot_reach: float  # horizontal, femur axis -> foot
    knee_reach: float  # horizontal, knee axis -> foot
    hip_height: float


def solve_pose(foot_radius: float, hip_height: float) -> Pose | None:
    """2-link IK in the leg's vertical plane (coxa already points the plane).

    foot_radius: horizontal distance femur-axis -> foot. Returns None if
    unreachable or if the knee would have to go below the hip (we keep the
    spidertron high-knee posture: knee above hip, always).
    """
    dx, dz = foot_radius, -hip_height
    d2 = dx * dx + dz * dz
    d = math.sqrt(d2)
    if d > FEMUR_LEN + TIBIA_LEN or d < abs(FEMUR_LEN - TIBIA_LEN):
        return None
    # angle at hip between the hip->foot line and the femur (law of cosines)
    cos_a = (FEMUR_LEN**2 + d2 - TIBIA_LEN**2) / (2 * FEMUR_LEN * d)
    a = math.acos(max(-1.0, min(1.0, cos_a)))
    base = math.atan2(dz, dx)
    femur_pitch = base + a  # knee-up solution
    if femur_pitch < math.radians(5.0):
        return None  # knee at/below hip: not this robot
    knee_x = FEMUR_LEN * math.cos(femur_pitch)
    knee_z = FEMUR_LEN * math.sin(femur_pitch)
    drop = math.atan2(knee_z + hip_height, dx - knee_x)
    return Pose(
        femur_pitch=femur_pitch,
        knee_drop=drop,
        foot_reach=foot_radius,
        knee_reach=dx - knee_x,
        hip_height=hip_height,
    )


# ---------------------------------------------------------------------------
# Torques
# ---------------------------------------------------------------------------


def stance_torques_kgcm(pose: Pose, foot_force_n: float) -> tuple[float, float, float]:
    """(coxa, femur, knee) torques in kg*cm for vertical foot force F."""
    nm_to_kgcm = 100.0 / G
    femur = foot_force_n * (pose.foot_reach / 1000.0) * nm_to_kgcm
    knee = foot_force_n * (pose.knee_reach / 1000.0) * nm_to_kgcm
    # Yaw: horizontal shear at the foot (propulsion/friction), 0.5 F, acting
    # at the full reach from the yaw axis.
    coxa = 0.5 * foot_force_n * ((COXA_LEN + pose.foot_reach) / 1000.0) * nm_to_kgcm
    return coxa, femur, knee


def swing_torques_kgcm() -> tuple[float, float]:
    """(femur, knee) gravity torque lifting the fully-extended horizontal leg.

    Femur pivot at 0: femur mass at L_f/2, knee servo at L_f, tibia mass at
    L_f + L_t/2 (tibia + foot). Worst case by construction.
    """
    lf, lt = FEMUR_LEN / 1000.0, TIBIA_LEN / 1000.0
    m_f, m_ks, m_t = PLA_FEMUR_G / 1000.0, KNEE_SERVO.mass_g / 1000.0, PLA_TIBIA_G / 1000.0
    nm_to_kgcm = 100.0 / G
    femur = G * (m_f * lf / 2 + m_ks * lf + m_t * (lf + lt / 2)) * nm_to_kgcm
    knee = G * (m_t * lt / 2) * nm_to_kgcm
    return femur, knee


def main() -> None:
    m = robot_mass_g() / 1000.0
    w = m * G
    print(f"leg mass        : {leg_mass_g():7.1f} g")
    print(f"robot mass      : {m * 1000:7.1f} g  (weight {w:.1f} N)")
    print(f"geometry        : coxa {COXA_LEN:.0f} / femur {FEMUR_LEN:.0f} / tibia {TIBIA_LEN:.0f} mm")
    print(f"hip height      : {HIP_HEIGHT:.0f} mm, body clearance {BODY_CLEAR:.0f} mm (obstacle spec 200 mm)")

    # Stance sweep: body heights the robot will actually use (crouch on rough
    # terrain to full stand) x foot radii from tucked to splayed.
    worst = {"coxa": (0.0, None, ""), "femur": (0.0, None, ""), "knee": (0.0, None, "")}
    cases = [("nominal W/3", w / 3.0), ("worst W/2", w / 2.0)]
    for hip_h in (HIP_MIN, 240.0, HIP_HEIGHT):
        for reach in range(int(REACH_MIN), int(REACH_MAX) + 1, 10):
            pose = solve_pose(float(reach), hip_h)
            if pose is None:
                continue
            for case, f in cases:
                c, fe, kn = stance_torques_kgcm(pose, f)
                for key, val in (("coxa", c), ("femur", fe), ("knee", kn)):
                    if case.startswith("worst") and val > worst[key][0]:
                        worst[key] = (val, pose, case)

    print(f"\n--- worst stance torques inside envelope (reach <= {REACH_MAX:.0f}, hip >= {HIP_MIN:.0f}, W/2) ---")
    for joint in ("coxa", "femur", "knee"):
        t, pose, _ = worst[joint]
        print(
            f"{joint:5s}: {t:5.1f} kg*cm  at reach {pose.foot_reach:.0f} mm, "
            f"hip {pose.hip_height:.0f} mm, femur {math.degrees(pose.femur_pitch):+.0f} deg"
        )

    # Nominal figures at the standing pose for the sustained-torque check.
    stand = solve_pose(270.0, HIP_HEIGHT)
    assert stand is not None
    c_n, f_n, k_n = stance_torques_kgcm(stand, w / 3.0)
    print("\n--- nominal standing pose (reach 270 mm, tripod W/3) ---")
    print(f"femur pitch {math.degrees(stand.femur_pitch):+.0f} deg, tibia drop {math.degrees(stand.knee_drop):.0f} deg")
    print(f"coxa {c_n:.1f} / femur {f_n:.1f} / knee {k_n:.1f} kg*cm")

    # Sustained standing on all six legs — the thermal case the "rated"
    # figure exists for.
    c_s, f_s, k_s = stance_torques_kgcm(stand, w / 6.0)
    print("\n--- sustained standing, all 6 legs (W/6) ---")
    print(f"coxa {c_s:.1f} / femur {f_s:.1f} / knee {k_s:.1f} kg*cm  (vs rated: check <= 1.0x)")

    fs, ks = swing_torques_kgcm()
    print("\n--- swing (leg extended horizontal, self-weight only) ---")
    print(f"femur {fs:.1f} / knee {ks:.1f} kg*cm")

    print("\n--- proposed servos vs requirements ---")
    rows = [
        ("coxa", COXA_SERVO, worst["coxa"][0], c_n),
        ("femur", FEMUR_SERVO, worst["femur"][0], f_n),
        ("knee", KNEE_SERVO, worst["knee"][0], k_n),
    ]
    print(f"{'joint':6s} {'servo':26s} {'worst':>7s} {'SF-stall':>9s} {'nominal':>8s} {'SF-rated':>9s}")
    for joint, servo, worst_t, nom_t in rows:
        print(
            f"{joint:6s} {servo.name:26s} {worst_t:6.1f}  {servo.stall_kgcm / worst_t:8.2f}  "
            f"{nom_t:7.1f}  {servo.rated_kgcm / nom_t:8.2f}"
        )

    n_coxa, n_pitch = 6, 12
    cost = n_coxa * COXA_SERVO.price_usd + n_pitch * FEMUR_SERVO.price_usd
    print(f"\nactuator cost   : 6x {COXA_SERVO.name} + 12x {FEMUR_SERVO.name} ~= ${cost:.0f}")

    # URDF actuator parameters implied by the selection (effort limit ~70% of
    # stall — same derating the XM430 baseline config used: 4.1 -> 3.0).
    print("\n--- URDF/sim actuator parameters (derated) ---")
    for joint, servo in (("coxa", COXA_SERVO), ("femur/knee", FEMUR_SERVO)):
        stall_nm = servo.stall_kgcm * G / 100.0
        no_load = math.radians(60.0) / servo.speed_s_per_60deg
        print(
            f"{joint:10s}: effort_limit {0.7 * stall_nm:4.2f} N*m (stall {stall_nm:.2f}), "
            f"velocity_limit {0.75 * no_load:4.2f} rad/s (no-load {no_load:.2f})"
        )
    print("\n--- all candidates, for reference (stall kg*cm / mass g / $) ---")
    for s in CANDIDATES:
        print(f"  {s.name:26s} {s.stall_kgcm:5.1f} / {s.mass_g:5.1f} / ${s.price_usd:.0f}")


if __name__ == "__main__":
    main()
