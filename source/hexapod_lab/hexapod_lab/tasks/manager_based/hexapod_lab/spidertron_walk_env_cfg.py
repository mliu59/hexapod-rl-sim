"""Walk task: follow a direction+speed command at a commanded hip height.

The merge point of the task family: velocity walking (M2 lineage) + height
tracking + gated stand behavior, one policy. Commands are two independent
terms:

* ``base_motion`` (mdp/commands.py DirectionSpeedCommand): world-frame target
  heading + forward speed, heading mode -- a P-controller turns heading error
  into a yaw-rate command, and forward speed is cos-projected so a robot facing
  away turns in place first. Strict facing, no strafing. ~10% of envs are
  commanded idle per resample.
* ``base_height``: the 0.18-0.35 m setpoint from the height-track task.

Mode gating (design discussion in docs/SPIDERTRON_TASKS.md): stand-still terms
(all feet down) apply only to idle envs; gait terms (air-time shaping) only to
moving-or-turning envs; everything else is mode-free. The speed-tracking reward
is attenuated as the commanded height leaves nominal, so at the range extremes
height wins over speed rather than trading against it.

Obs: 69 dims = base 63 + motion command 4 + height command 1 + height error 1.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .spidertron_base_env_cfg import (
    FOOT_TIP_OFFSET,
    NOMINAL_HEIGHT,
    SpidertronBaseEnvCfg,
)
from .spidertron_base_env_cfg import ObservationsCfg as BaseObservationsCfg
from .spidertron_base_env_cfg import TerminationsCfg as BaseTerminationsCfg
from .spidertron_height_env_cfg import HEIGHT_RANGE

# Froude-scaled from the M2 analysis: the servos bind before 0.43x of ANYmal's
# speed does, so start conservative and raise once tracking is real.
MAX_SPEED = 0.3  # m/s
MAX_YAW_RATE = 0.8  # rad/s
# v10 (march finding): band aligned with the plant's revealed ~0.14 s natural
# swing cadence. The v4 escalation (0.4 target / 0.15 floor) sat entirely
# above it and made air-time fight the duty term; the aligned band let both
# climb together on the march task (duty_min 0.35, tracking 93%).
TARGET_SWING_TIME = 0.25  # s
MIN_AIR_TIME = 0.08  # s
FOOT_CLEARANCE = 0.04  # m swing-foot lift target


@configclass
class CommandsCfg:
    base_motion = mdp.DirectionSpeedCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        heading_control_stiffness=0.5,
        max_yaw_rate=MAX_YAW_RATE,
        rel_standing_envs=0.1,
        debug_vis=False,
        ranges=mdp.DirectionSpeedCommandCfg.Ranges(heading=(-math.pi, math.pi), speed=(0.0, MAX_SPEED)),
    )
    base_height = mdp.UniformHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        debug_vis=False,
        ranges=mdp.UniformHeightCommandCfg.Ranges(height=HEIGHT_RANGE),
    )


@configclass
class ObservationsCfg(BaseObservationsCfg):
    """Base proprioception + motion command + height command + height error."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_motion"})
        height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_height"})
        height_error = ObsTerm(func=mdp.height_setpoint_error, params={"command_name": "base_height"})
        # v9: binary foot contacts (6) -- the gait rewards key on contact
        # events, so the policy must be able to observe them
        foot_contacts = ObsTerm(
            func=mdp.foot_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia")},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class WalkRewardsCfg:
    """Tracking + mode-gated gait/stand shaping + stability + regularizers."""

    # -- tracking (the positive terms)
    track_forward_vel = RewTerm(
        func=mdp.track_forward_vel_hgated_exp,
        weight=2.0,
        params={
            "std": 0.5 * MAX_SPEED,
            "command_name": "base_motion",
            "height_command_name": "base_height",
            "nominal_height": NOMINAL_HEIGHT,
            # attenuation ~0.5 at the ends of the 0.18-0.35 m command range
            "height_std": 0.09,
        },
    )
    # v9: merged the yaw-rate tracking term into this one (the yaw-rate command
    # is a deterministic P-function of heading error, so the pair rewarded the
    # same channel at two derivatives); combined weight here. Cosine kernel has
    # gradient everywhere on the circle (v3 lesson).
    track_heading = RewTerm(
        func=mdp.track_heading_cos,
        weight=2.5,
        params={"command_name": "base_motion"},
    )
    base_height_exp = RewTerm(
        func=mdp.base_height_command_exp,
        weight=2.0,
        params={"command_name": "base_height", "std": 0.03},
    )

    # -- gait shaping, gated on moving-or-turning
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_target_active,
        # M2 lesson: lifting a leg must be worth more than holding still, or
        # the gait terms sum negative and the optimum is to freeze.
        # v5: 4.0 -> 6.0 -- v4 stalled in a suppressed-shuffle equilibrium
        # (feet_air_time flat at ~-0.02 for 600 iters while tracking converged)
        weight=6.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "command_name": "base_motion",
            "target_time": TARGET_SWING_TIME,
            "min_air_time": MIN_AIR_TIME,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_active,
        # v7: -0.25 -> -0.1 -- at -0.25 this was one of the largest penalties
        # and taxed stepping EXPLORATION (irregular early steps cost variance
        # immediately, long before a regular gait pays), helping pin the
        # skate-shuffle equilibrium. It is a polish regularizer, not a driver.
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "command_name": "base_motion",
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_tibia"),
            "target_height": FOOT_CLEARANCE,
            "tip_offset": FOOT_TIP_OFFSET,
        },
    )
    # v8: pays the stride itself -- forward speed x honestly-planted feet.
    # Neither skating (feet fail slip_tol) nor standing (zero speed) collects.
    # v10: weight 0.5 -> 1.0 (march value; the honest-stride channel earned it)
    stance_progress = RewTerm(
        func=mdp.stance_progress,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_tibia"),
            "slip_tol": 0.05,
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        # v7: -0.5 -> -2.0 -- v6 showed the policy moving at commanded speed by
        # SLIDING planted feet (slip cost -0.13/episode was cheaper than
        # learning to step). Skating must cost more than stepping.
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_tibia"),
        },
    )

    # -- stand behavior, gated on idle
    feet_off_ground = RewTerm(
        func=mdp.feet_off_ground_idle,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "command_name": "base_motion",
        },
    )
    # v9 anti-hop: charges every step for feet held airborne > 1 s while
    # moving -- the tripod-hop carried curled legs that never landed, which
    # neither the idle-gated term above nor the touchdown-priced air-time
    # floor could see
    feet_airborne_too_long = RewTerm(
        func=mdp.feet_airborne_too_long,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "command_name": "base_motion",
            "max_air_time": 1.0,
        },
    )
    # v10 (march finding): load coupling -- every foot's contact duty in
    # [0.45, 0.75] while moving. The term that finally made all six legs
    # carry load on the march task (duty_min 0.02 -> 0.35).
    foot_duty = RewTerm(
        func=mdp.foot_duty_deviation,
        weight=-4.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "command_name": "base_motion",
            "target": 0.6,
            "tol": 0.15,
        },
    )
    # v9 anti-dithering: prices time-at-saturation (top decile of effort) --
    # the v8 bang-bang style held joints at the torque limit ~95% of steps,
    # invisible to dof_torques_l2 (magnitude is capped at the limit)
    torque_saturation = RewTerm(
        func=mdp.torque_saturation,
        weight=-5.0,
        params={"threshold": 0.9},
    )

    # -- torso stability
    hip_height_variance = RewTerm(
        func=mdp.hip_height_variance,
        # hips differing by ~1 cm -> var ~1e-4 -> costs 0.01/step at this weight
        weight=-100.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_coxa")},
    )
    # v9 audit: flat_orientation_l2 dropped -- it triple-covered orientation
    # with hip_height_variance and ang_vel_xy_l2
    lat_vel_l2 = RewTerm(func=mdp.base_lin_vel_y_l2, weight=-2.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    base_ang_acc_l2 = RewTerm(
        func=mdp.base_ang_acc_l2,
        weight=-2.5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )

    # -- regularizers (v2-height-track values)
    # v9 audit: joint_deviation_l1 dropped -- anchoring to the standing pose
    # fights stride length and crouch/stilt poses; dof_pos_limits still guards
    # the workspace edges
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    # v9: x20 -- at -0.01 command thrash was effectively free and the policy
    # controlled through saturation-averaged dithering (v8 post-mortem)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.2)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_femur", ".*_coxa"]),
            "threshold": 1.0,
        },
    )


@configclass
class WalkCurriculumCfg:
    """v10: the v8 speed ramp is gone (falsified -- the discovery barrier was
    control style, not speed). This group now carries the march task's live
    gait-structure probes instead."""

    metric_foot_duty_min = CurrTerm(
        func=mdp.foot_duty_metric, params={"sensor_name": "contact_forces", "reduce": "min"}
    )
    metric_foot_duty_mean = CurrTerm(
        func=mdp.foot_duty_metric, params={"sensor_name": "contact_forces", "reduce": "mean"}
    )
    # harness-level friction-evasion tripwire (march-free2 post-mortem)
    metric_foot_slip = CurrTerm(func=mdp.foot_slip_metric, params={"sensor_name": "contact_forces"})


@configclass
class WalkTerminationsCfg(BaseTerminationsCfg):
    # end hopeless rollouts early instead of collecting 12 s of a tipped robot
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class SpidertronWalkEnvCfg(SpidertronBaseEnvCfg):
    """Spidertron walking a commanded direction/speed at a commanded height."""

    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: WalkRewardsCfg = WalkRewardsCfg()
    terminations: WalkTerminationsCfg = WalkTerminationsCfg()
    curriculum: WalkCurriculumCfg = WalkCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # room for walk-stop-walk transitions within one episode (~2-3 command
        # windows per term)
        self.episode_length_s = 12.0
        # v7: wider stride room WITH the exploration product kept safe. v5's
        # 0.35 scale at init_noise_std 1.0 collapsed every env at spawn
        # (product 0.35 rad of early joint swing saturates the knees); the
        # runner cfg pairs this 0.35 with init_noise_std 0.7 -> product 0.245,
        # at or below the proven-survivable v4 level (0.25 x 1.0).
        self.actions.joint_pos.scale = 0.35


@configclass
class SpidertronWalkEnvCfg_PLAY(SpidertronWalkEnvCfg):
    """Small, deterministic version for watching rollouts."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
