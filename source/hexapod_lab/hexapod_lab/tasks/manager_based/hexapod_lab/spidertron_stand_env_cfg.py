"""Stand-still task for the spidertron: hold the base at the nominal hip height.

The trivial first task on the new robot, and the worked example of the task
pipeline (see ``spidertron_base_env_cfg.py``): everything shared lives in the
base config, this file only defines what standing *is* -- a positive reward for
being at height, penalties for moving, tipping, or leaning on the wrong bodies.

No commands: the policy's only job is to stay put. Success looks like
``base_height_exp`` saturating near its weight and episode length pinned at
time-out.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .spidertron_base_env_cfg import NOMINAL_HEIGHT, SpidertronBaseEnvCfg


@configclass
class StandRewardsCfg:
    """Hold height, stay level, stay still, stand on feet only."""

    # -- task: the one positive term the policy climbs
    base_height_exp = RewTerm(
        func=mdp.base_height_target_exp,
        weight=2.0,
        # std 0.03: full credit within ~3 cm of the 0.2745 m hip height,
        # ~zero credit when collapsed (0.05-0.10 m, per check_collapse.py)
        params={"target_height": NOMINAL_HEIGHT, "std": 0.03},
    )

    # -- stay still and level
    lin_vel_xy_l2 = RewTerm(func=mdp.base_lin_vel_xy_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    # -- regularizers, carried over from the M2 scaling analysis (3 N*m-class
    #    torques need ~700x ANYmal's torque penalty weight; 18 DOF inflates
    #    joint sums 1.5x)
    joint_deviation_l1 = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
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
class SpidertronStandEnvCfg(SpidertronBaseEnvCfg):
    """Spidertron standing at nominal hip height on flat ground."""

    rewards: StandRewardsCfg = StandRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # standing needs no long horizon; shorter episodes give more resets and
        # more randomized initial states per unit compute
        self.episode_length_s = 5.0


@configclass
class SpidertronStandEnvCfg_PLAY(SpidertronStandEnvCfg):
    """Small, deterministic version for watching rollouts."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
