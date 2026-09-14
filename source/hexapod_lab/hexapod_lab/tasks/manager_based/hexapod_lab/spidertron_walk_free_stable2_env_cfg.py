"""Walk-free-stable v3: no height task, yaw-free wobble price, finer physics.

Third variation on pricing wobble in the free walk task (lineage:
``spidertron_walk_free_stable_env_cfg.py`` v1 froze the robot at walk-task
weights, v2 at 10x-reduced weights walked with halved slip and best-yet
antiphase but degraded heading tracking). Changes from v2, all aimed at the
residual deficit -- turning:

* **Height task removed.** No ``base_height`` command, no height reward, no
  height terms in the observation (march-task obs: proprioception + motion
  command + binary foot contacts, 73 dims). Forward-speed tracking loses its
  height gate (plain ``track_forward_vel_exp``, fixed std 0.3). NOTE: nothing
  anchors ride height now except terminations and physics -- where the body
  settles is a free choice of the optimizer and part of the measurement.
* **Wobble price is yaw-free and halved.** ``base_ang_acc_xy_l2`` prices only
  roll/pitch acceleration -- yaw acceleration is what turning is, and v1/v2
  showed the all-axis charge bites heading tracking first. All wobble weights
  halved from v2 (hip_var -5, lin_vel_z/ang_vel_xy -0.025, ang_acc_xy
  -1.25e-5): still a bill, deliberately lighter.
* **Finer physics, longer episodes.** sim.dt 0.005 -> 0.004 (250 Hz physics,
  125 Hz control at decimation 2) and episode 12 s -> 22 s (~4 command
  windows: more turn transitions per episode in-distribution).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .spidertron_march_env_cfg import MarchMetricsCfg, ObservationsCfg
from .spidertron_walk_env_cfg import WalkTerminationsCfg
from .spidertron_walk_free_env_cfg import SpidertronWalkFreeEnvCfg
from .spidertron_walk_free_stable_env_cfg import WalkFreeStableRewardsCfg


@configclass
class WalkFreeStable2RewardsCfg(WalkFreeStableRewardsCfg):
    """v2 rewards, height-free and with the yaw-free halved wobble price.

    The height command/reward themselves are deleted in the env cfg's
    ``__post_init__`` (the None-ing pattern used for events.push_robot).
    """

    # no height command to gate on: plain exp tracking, same fixed std
    track_forward_vel = RewTerm(
        func=mdp.track_forward_vel_exp,
        weight=2.0,
        params={"std": 0.3, "command_name": "base_motion"},
    )

    # wobble bill, halved from v2 and yaw-free
    hip_height_variance = RewTerm(
        func=mdp.hip_height_variance,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_coxa")},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.025)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.025)
    base_ang_acc_l2 = RewTerm(
        func=mdp.base_ang_acc_xy_l2,
        weight=-1.25e-5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )


@configclass
class SpidertronWalkFreeStable2EnvCfg(SpidertronWalkFreeEnvCfg):
    """Heading+speed only, yaw-free wobble bill, 250 Hz physics, 22 s episodes."""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: WalkFreeStable2RewardsCfg = WalkFreeStable2RewardsCfg()
    terminations: WalkTerminationsCfg = WalkTerminationsCfg()
    curriculum: MarchMetricsCfg = MarchMetricsCfg()

    def __post_init__(self):
        super().__post_init__()
        # the height task is gone: no command term, no height reward
        self.commands.base_height = None
        self.rewards.base_height_exp = None
        # finer physics: 250 Hz sim, 125 Hz control (decimation 2 inherited)
        self.sim.dt = 0.004
        # ~4 command windows per episode instead of ~2-3
        self.episode_length_s = 22.0


@configclass
class SpidertronWalkFreeStable2EnvCfg_PLAY(SpidertronWalkFreeStable2EnvCfg):
    """Small, deterministic version for watching rollouts."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
