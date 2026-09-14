"""Single source of truth for the spidertron hexapod's parameters.

Everything downstream — link solids, STL meshes, URDF joints/limits/inertials
— derives from this module. To iterate the robot, edit here and rerun
``scripts/build_robot.py``; never edit the generated URDF.

Sign conventions (match the legacy placeholder generator so the Isaac Lab
configs port unchanged):
  coxa  : yaw about +z, 0 = radial out from body, CCW positive
  femur : pitch about +y, NEGATIVE = lift (standing pose is -30 deg)
  tibia : knee flexion about +y, 0 = tibia aligned with femur extension,
          positive folds the foot down/back; angle = 180 deg - interior
Derived: standing interior angle 76 deg -> knee angle +104 deg.

Units: millimetres and degrees here; the URDF emitter converts to m/rad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ServoSpec:
    """A serial-bus servo, numbers from datasheet + vendor CAD probing."""

    name: str
    mass_g: float  # datasheet (vendor CAD volume lies; see PROVENANCE)
    stall_kgcm: float
    effort_limit_nm: float  # derated ~70% of stall, enforced in sim
    velocity_limit_rad_s: float  # ~75% of no-load speed
    # case envelope, verified against official STEP (see servo_lp in links.py)
    case_l: float = 45.2
    case_w: float = 24.7
    case_h: float = 35.7
    shaft_from_rear: float = 10.1  # vendor CAD; datasheet drawing says 11.25
    horn_r: float = 10.0
    horn_h: float = 1.7
    step_file: str = ""  # vendor ground truth, for mass-distribution


STS3215 = ServoSpec(
    name="STS3215",
    mass_g=60.0,
    stall_kgcm=30.0,
    effort_limit_nm=2.06,
    velocity_limit_rad_s=5.38,
    step_file="vendor/sts3215/STS3215.step",
)
STS3250 = ServoSpec(
    name="STS3250",
    mass_g=74.5,
    stall_kgcm=50.0,
    effort_limit_nm=3.43,
    velocity_limit_rad_s=5.91,
    step_file="vendor/sts3250/STS3250.step",
)


@dataclass(frozen=True)
class HexapodParams:
    # --- leg geometry (mm) ---
    coxa_len: float = 60.0  # yaw axis -> femur pitch axis
    femur_len: float = 190.0  # femur axis -> knee axis
    tibia_len: float = 380.0  # knee axis -> foot tip

    # --- body (mm) ---
    body_across_flats: float = 160.0
    body_h: float = 80.0
    hip_r: float = 95.0  # yaw axes on this circle, at hex vertices
    dome_r: float = 46.0
    hip_bay_r: float = 34.0  # clearance wells carved for servo + bracket

    # --- stance / posture ---
    hip_height: float = 270.0  # standing; body underside at hip_height-body_h/2
    stand_pitch_deg: float = 30.0  # femur above horizontal (joint = -30 deg)
    stand_interior_deg: float = 76.0  # knee interior (joint = +104 deg)

    # --- joint limits (deg), collision-verified at extremes (DESIGN_LOG) ---
    coxa_range_deg: tuple = (-25.0, 25.0)
    femur_lift_range_deg: tuple = (-25.0, 85.0)  # physical lift; joint = -lift
    knee_interior_range_deg: tuple = (35.0, 150.0)

    # --- printed-part target masses (g): budget values, geometry pinned to them ---
    mass_bracket_g: float = 30.0
    mass_femur_g: float = 60.0
    mass_tibia_g: float = 85.0
    mass_body_shell_g: float = 220.0
    mass_electronics_g: float = 380.0  # 3S pack + controller + wiring lump

    # --- actuators ---
    servo_yaw: ServoSpec = STS3215
    servo_pitch: ServoSpec = STS3250  # femur and knee

    # --- structure details (mm), from the verified session geometry ---
    bracket_w: float = 30.0
    bracket_t: float = 6.0
    plate_gap_inner: float = 13.35  # femur plate inner faces at +/- this
    plate_t: float = 4.0
    plate_h: float = 30.0
    tibia_arm_w: float = 10.0  # single-side arm, y in [1,11]
    tibia_arm_root_h: float = 26.0
    tibia_arm_tip_h: float = 7.0
    tibia_hub_r: float = 14.0
    tibia_hub_t: float = 12.0

    # --- naming: leg id -> hip azimuth (deg CCW from +x forward) ---
    legs: dict = field(
        default_factory=lambda: {
            "LF": 30.0,
            "LM": 90.0,
            "LR": 150.0,
            "RR": 210.0,
            "RM": 270.0,
            "RF": 330.0,
        }
    )

    # ------------------------------------------------------------------
    @property
    def stand_knee_deg(self) -> float:
        return 180.0 - self.stand_interior_deg

    @property
    def knee_range_deg(self) -> tuple[float, float]:
        lo_i, hi_i = self.knee_interior_range_deg
        return (180.0 - hi_i, 180.0 - lo_i)  # (30, 145)

    @property
    def femur_joint_range_deg(self) -> tuple[float, float]:
        lo, hi = self.femur_lift_range_deg
        return (-hi, -lo)  # (-85, +25); negative = up

    def default_joint_pose_rad(self) -> dict[str, float]:
        """Per-joint-type standing pose, radians (sync with robots/hexapod.py)."""
        return {
            "coxa": 0.0,
            "femur": -math.radians(self.stand_pitch_deg),
            "tibia": math.radians(self.stand_knee_deg),
        }

    def foot_z_standing(self) -> float:
        """FK sanity: foot height below hip plane at the standing pose (mm)."""
        up = math.radians(self.stand_pitch_deg)
        drop = math.radians(180.0 - self.stand_pitch_deg - self.stand_interior_deg)
        return self.femur_len * math.sin(up) - self.tibia_len * math.sin(drop)


PARAMS = HexapodParams()
