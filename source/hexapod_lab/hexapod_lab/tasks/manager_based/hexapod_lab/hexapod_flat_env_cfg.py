"""Velocity-tracking task for the 18-DOF hexapod on flat ground (milestone M2).

Adapted from Isaac Lab's ``LocomotionVelocityRoughEnvCfg`` (the config behind the
ANYmal-C rough baseline), but written out in full rather than subclassed. The
structure of that task is morphology-agnostic and transfers directly; almost none
of its *numbers* do, and inheriting them silently would hide that. See the notes
on each block for what was rescaled and why.

Scale, hexapod vs ANYmal-C: 3 kg vs ~50 kg, 0.111 m vs 0.6 m base height, 3 N*m
vs 80 N*m joints, 4.8 rad/s vs 7.5 rad/s. In dimensionless terms the hexapod is
~3.4x stronger and ~3.6x slower for its size -- "strong but slow" -- and it is
statically stable on a tripod. That combination, not the raw ratios, drives the
choices below.

Body naming (from scripts/generate_hexapod_urdf.py): base_link, and per leg
{L,R}{F,M,R}_{coxa,femur,tibia}. The contact point is the tibia tip; there is no
separate foot link.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from hexapod_lab.robots import HEXAPOD_CFG

from . import mdp

##
# Scale constants -- derived in one place so the reasoning is visible
##

# Froude-scaled speed: v ~ sqrt(g*l) gives 1.04 m/s for the hexapod vs 2.43 m/s
# for ANYmal, i.e. 0.43x. ANYmal's +/-1.0 m/s maps to ~0.43 m/s, and a
# body-lengths/s check gives ~0.32 m/s. The servos bind before either, so start
# conservative and raise once the policy is actually tracking.
MAX_LIN_VEL_X = 0.3  # m/s
MAX_LIN_VEL_Y = 0.15  # m/s
MAX_ANG_VEL_Z = 0.8  # rad/s

# Standing height from scripts/check_robot.py (foot 0.104 m below the hip).
NOMINAL_HEIGHT = 0.111  # m

# Target swing duration per leg. A tripod gait at 2-3 Hz puts swing around
# 0.17-0.25 s; 4.8 rad/s joints cannot do much better.
TARGET_SWING_TIME = 0.2  # s

# Swing-foot clearance target. ~30% of the 0.104 m ground clearance: enough to
# stop the policy scuffing, not so much that it wastes the servo speed budget.
FOOT_CLEARANCE = 0.03  # m


##
# Scene
##


@configclass
class HexapodFlatSceneCfg(InteractiveSceneCfg):
    """Flat ground, one hexapod per env, contact sensing on every body."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot = HEXAPOD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # every body, so undesired-contact terms can name femurs and the chassis
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # no height scanner: flat ground in M2, and the scandot design is deliberately
    # deferred to M3 (a 0.1 m grid would give 3 samples across a 0.30 m body anyway)

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##
# MDP
##


@configclass
class CommandsCfg:
    """Commanded base velocity.

    Resampling every 5 s rather than ANYmal's 10 s: the episode is shorter in the
    same proportion, so the policy still sees ~2 commands per episode.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-MAX_LIN_VEL_X, MAX_LIN_VEL_X),
            lin_vel_y=(-MAX_LIN_VEL_Y, MAX_LIN_VEL_Y),
            ang_vel_z=(-MAX_ANG_VEL_Z, MAX_ANG_VEL_Z),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Joint position targets, offset from the default standing pose.

    scale 0.25 rad against ANYmal's 0.5: the coxa range is only +/-0.9 rad, and
    a full-scale action should not be able to demand more than the servos can
    track within one control step.
    """

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Proprioception only -- 66 dims (3+3+3+3+18+18+18)."""

    @configclass
    class PolicyCfg(ObsGroup):
        # noise magnitudes are scaled to this robot: ANYmal's +/-1.5 rad/s joint
        # velocity noise is a third of the hexapod's entire 4.8 rad/s range
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Domain randomization, scaled to a 3 kg robot."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # +/-0.3 kg on a 1.2 kg chassis, the same ~10% of body mass ANYmal
            # randomizes with its +/-5 kg
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.3, 0.3),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            # scaled from ANYmal's +/-0.05 m by the 0.3/0.93 body-length ratio
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.005, 0.005)},
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-math.pi, math.pi)},
            # no z offset: the robot must start at standing height. Dropping it
            # from higher saturates the 3 N*m tibias on impact and it cannot get up.
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            # ANYmal's (0.5, 1.5) would put the tibia anywhere in 0.87-2.6 rad and
            # start a large fraction of envs already collapsed
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )


@configclass
class RewardsCfg:
    """Reward terms.

    Weights are a first pass. Two rescalings were needed relative to the ANYmal
    config and both are easy to get wrong:

    * terms that sum over joints or feet grow with 18 DOF / 6 legs instead of
      12 / 4, inflating them ~1.5x relative to the fixed-weight tracking term;
    * terms quadratic in a physical quantity scale with that quantity squared.
      Torque is the big one: at 3 N*m instead of 80 N*m, ANYmal's -1e-5 torque
      penalty would be ~700x too weak to regularize anything.

    Check the per-term breakdown in TensorBoard after the first run rather than
    trusting these numbers.
    """

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        # std tracks the command range: ANYmal uses 0.5 on +/-1.0 commands
        params={"command_name": "base_velocity", "std": 0.5 * MAX_LIN_VEL_X},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5 * MAX_ANG_VEL_Z},
    )

    # -- gait shaping (hexapod-specific, see mdp/rewards.py)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_target,
        # 4.0, not 1.0: at weight 1.0 the 60-iteration smoke run had air-time
        # paying +0.014 against a -0.059 variance penalty, so the gait terms
        # summed negative and the cheapest way to satisfy them was to stop
        # stepping. Lifting a leg has to be worth more than holding still.
        weight=4.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "command_name": "base_velocity",
            "target_time": TARGET_SWING_TIME,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance,
        # a regularizer, not a driver. Note the degenerate optimum: a robot that
        # never lifts a leg has zero air-time variance, so this term must stay
        # well below the air-time reward or it pays for standing still.
        weight=-0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia")},
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_tibia"),
            "target_height": FOOT_CLEARANCE,
        },
    )

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-3)
    # halved from ANYmal's -2.5e-7: with 18 joints at 100 Hz this was the single
    # largest penalty in the smoke run (-0.26 against +0.59 of tracking)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        # a sprawled hexapod can crouch and shuffle to track velocity; this keeps
        # it standing at the height the leg geometry was designed around
        params={"target_height": NOMINAL_HEIGHT},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_femur", ".*_coxa"]),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )


##
# Environment
##


@configclass
class HexapodFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Hexapod velocity tracking on flat ground."""

    scene: HexapodFlatSceneCfg = HexapodFlatSceneCfg(num_envs=4096, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # 100 Hz control, not ANYmal's 50 Hz. The hexapod's characteristic time
        # sqrt(l/g) is 0.106 s against ANYmal's 0.247 s, so its dynamics run ~2.3x
        # faster and a 50 Hz policy is correspondingly coarser.
        self.decimation = 2
        # 8 s ~ 20 s on ANYmal after the same 0.43x time scaling
        self.episode_length_s = 8.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class HexapodFlatEnvCfg_PLAY(HexapodFlatEnvCfg):
    """Small, deterministic version for watching rollouts."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
