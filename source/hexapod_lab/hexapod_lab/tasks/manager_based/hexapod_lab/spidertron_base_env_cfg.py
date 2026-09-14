"""Task-agnostic base config for spidertron RL tasks.

This is the reusable half of the training pipeline: everything a task shares
regardless of its objective — the scene (robot, flat ground, contact sensing),
proprioceptive observations, joint-position actions, domain randomization, and
sim timing. A new task is a subclass that adds what defines it:

* a ``rewards`` configclass (custom terms go in ``mdp/rewards.py``),
* optionally a ``commands`` configclass and a command observation term,
* optionally extra terminations,
* a runner config in ``agents/rsl_rl_ppo_cfg.py`` and a ``gym.register`` entry
  in ``__init__.py``.

``scripts/rsl_rl/train.py --task <id>`` then trains it under the observability
scaffold with no further wiring. ``spidertron_stand_env_cfg.py`` is the worked
example.

Numbers here are spidertron-specific (Feetech STS servos, 0.2745 m hip height);
see ``robots/spidertron.py`` and ``cad/DESIGN_LOG.md`` for their derivation.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from hexapod_lab.robots import SPIDERTRON_CFG

from . import mdp

# Base (hip-plane) height when standing: feet reach 0.2745 m below the base
# origin at the standing pose (robots/spidertron.py). Tasks that care about
# height reference this instead of re-deriving it.
NOMINAL_HEIGHT = 0.2745  # m

# Foot contact sphere sits 0.377 m along the tibia +x axis (URDF collision
# geometry); at the standing pose the feet form a hexagon of this radius
# around the base origin. Keep in sync with the CAD emitter.
FOOT_TIP_OFFSET = 0.377  # m
FOOT_RADIUS = 0.424  # m


##
# Scene
##


@configclass
class SpidertronSceneCfg(InteractiveSceneCfg):
    """Flat ground, one spidertron per env, contact sensing on every body."""

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

    robot = SPIDERTRON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # every body, so undesired-contact terms can name femurs and the chassis.
    # Foot-ground contact is treated as BINARY: force_threshold is the single
    # place the boolean is derived (sensor's contact/air-time state machine);
    # task terms read contact state, never force magnitudes -- matching the
    # contact-switch feet a real build would have.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##
# Shared MDP pieces
##


@configclass
class ActionsCfg:
    """Joint position targets, offset from the default standing pose.

    scale 0.25 rad: a full-scale action should not demand more than the STS
    servos can track within one control step (5.4-5.9 rad/s at 100 Hz).
    """

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Proprioception only -- 63 dims (3+3+3+18+18+18).

    Commanded tasks extend ``PolicyCfg`` with a command term in their subclass.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Domain randomization. Magnitudes follow the M2 reasoning, rescaled where mass-dependent."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # Rubber/TPU-foot realism (raised from 0.6-1.0 / 0.4-0.8 after
            # free3: effective dynamic friction of 0.4-0.8 made sliding cheap
            # and the free optimum a shuffle). Rubber on ground is mu ~1.0-1.5;
            # higher static widens the stiction cone (feet hold under lateral
            # load), higher dynamic prices any slide that still happens.
            "static_friction_range": (1.0, 1.4),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # multiplicative so it stays ~10% of chassis mass without hardcoding
            # the CAD-derived value here
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-math.pi, math.pi)},
            # no z offset: spawn at standing height -- dropping saturates the
            # 3.4 N*m knees on impact (same lesson as the placeholder robot)
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
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )


##
# Base environment
##


@configclass
class SpidertronBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Common spidertron env config. Subclass and add ``rewards`` (and optionally
    ``commands``) to make a task."""

    scene: SpidertronSceneCfg = SpidertronSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # 100 Hz control (decimation 2 at dt 0.005), same reasoning as M2: the
        # characteristic time sqrt(l/g) is ~0.17 s, faster dynamics than a
        # 50 Hz policy resolves well.
        self.decimation = 2
        self.episode_length_s = 8.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
