"""Walk-free task: consolidated command-following with minimal priors, no modes.

The successor to the walk task after the free-run exploration
(docs/SIM_PHYSICS_EXPLOITS.md): the mode gating and the entire hand-built gait
suite are gone, and one policy covers stand / walk / height change as regions
of a single command space. Design rationale (session 2026-09-14):

* **No gating.** The gated terms existed because gait-shaping rewards
  misbehave across command modes (air-time at zero command rewards marching
  in place). With the gait suite deleted, no surviving term is
  mode-dependent: speed tracking at command 0 *is* the stand task.
  ``rel_standing_envs`` remains as command sampling only.
* **No gait terms.** Rewards are three tracking channels + survival + the
  hardware-feasibility suite. Gait structure is measured (duty / antiphase /
  slip metrics), never rewarded. Expectation, stated up front: within the
  1.7 m/s command cap -- free3's proven shuffle envelope -- the likely
  optimum is a command-following shuffle; the metrics tell us if anything
  better emerges. Raising the cap past 1.7 is the reserved emergence lever.
* **Energy in, ride quality out.** ``joint_power_positive`` (the mechanical
  half of the servo electrical bill) makes stillness the optimum at zero
  command without gating, and quietly prices the shuffle's Coulomb dragging
  bill (free4b measured it: ~15 W). The torso-stability block
  (hip_height_variance, lin_vel_z, ang_vel_xy, base_ang_acc) and lat_vel_l2
  are dropped as behavior priors: body motion is policed only by tracking
  error, terminations, and the energy meter.

Physics enforcement (DCMotor envelope, contact integrity, rubber friction) is
untouched -- rewards express preferences, physics violations get fixed in
physics. Obs stay blind: proprioception + commands + binary foot contacts.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .spidertron_base_env_cfg import NOMINAL_HEIGHT, SpidertronBaseEnvCfg
from .spidertron_height_env_cfg import HEIGHT_RANGE
from .spidertron_march_env_cfg import MarchMetricsCfg
from .spidertron_walk_env_cfg import ObservationsCfg, WalkTerminationsCfg

# Command cap = the max speed honest physics has demonstrated (free3 shuffle
# ceiling 1.7 m/s; the constrained max2 runner reached 2.13). Every command in
# range is physically satisfiable, so tracking failure is always the policy's
# fault, not the plant's.
MAX_SPEED = 1.7  # m/s
MAX_YAW_RATE = 0.8  # rad/s


@configclass
class CommandsCfg:
    """Walk's two command terms, speed range widened to the honest envelope."""

    base_motion = mdp.DirectionSpeedCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        heading_control_stiffness=0.5,
        max_yaw_rate=MAX_YAW_RATE,
        # ~10% of envs commanded to stand -- command sampling, not reward
        # gating: no term reads the idle flag
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
class WalkFreeRewardsCfg:
    """Tracking + survival + hardware feasibility. Nothing else."""

    # -- task tracking (the only positive terms)
    track_forward_vel = RewTerm(
        func=mdp.track_forward_vel_hgated_exp,
        weight=2.0,
        params={
            # fixed std, NOT 0.5*MAX_SPEED: at a 1.7 m/s range that would be
            # 0.85 -- uselessly loose. 0.3 keeps gradient at low speeds and
            # makes high-command shuffle shortfall actually cost.
            "std": 0.3,
            "command_name": "base_motion",
            "height_command_name": "base_height",
            "nominal_height": NOMINAL_HEIGHT,
            "height_std": 0.09,
        },
    )
    track_heading = RewTerm(func=mdp.track_heading_cos, weight=2.5, params={"command_name": "base_motion"})
    base_height_exp = RewTerm(
        func=mdp.base_height_command_exp,
        weight=2.0,
        params={"command_name": "base_height", "std": 0.03},
    )

    # -- survival: at 1.7 m/s commands a crash-prone sprint style can still
    # pay in tracking; dying must dominate it
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)

    # -- hardware feasibility (mode-free; the servos, not a gait opinion)
    torque_saturation = RewTerm(func=mdp.torque_saturation, weight=-5.0, params={"threshold": 0.9})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.2)
    brief_contacts = RewTerm(
        func=mdp.brief_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "min_contact_time": 0.1,
        },
    )
    # energy: positive mechanical power (W). Bill, not wall -- see the
    # function docstring for the calibration argument.
    joint_power = RewTerm(func=mdp.joint_power_positive, weight=-3.0e-3)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
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
class SpidertronWalkFreeEnvCfg(SpidertronBaseEnvCfg):
    """One consolidated policy: stand, walk to 1.7 m/s, track height -- no modes."""

    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: WalkFreeRewardsCfg = WalkFreeRewardsCfg()
    terminations: WalkTerminationsCfg = WalkTerminationsCfg()
    # gait structure is measured, not rewarded: duty, antiphase, slip tripwire
    curriculum: MarchMetricsCfg = MarchMetricsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ~2-3 command windows per episode: transitions are in-distribution
        self.episode_length_s = 12.0
        # v9 exploration pairing (scale x init_noise_std 0.7 = 0.245 rad,
        # at the proven-survivable level; runner cfg holds the 0.7)
        self.actions.joint_pos.scale = 0.35


@configclass
class SpidertronWalkFreeEnvCfg_PLAY(SpidertronWalkFreeEnvCfg):
    """Small, deterministic version for watching rollouts."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
