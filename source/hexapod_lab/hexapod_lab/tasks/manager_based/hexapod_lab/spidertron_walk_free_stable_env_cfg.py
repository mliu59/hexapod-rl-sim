"""Walk-free + wobble price: the free task with the torso-stability block back.

A/B against ``Spidertron-WalkFree-v0`` (identical commands, obs, physics,
tracking and hardware-feasibility rewards): the one change is reintroducing
the torso-stability penalties that the free rebuild deliberately dropped as
behavior priors. The question this run answers: does pricing wobble (torso
oscillation) merely damp the body while keeping the slide-shuffle, or does it
tip the optimum toward discrete stepping? Gait structure stays measured, not
rewarded -- the march metrics (duty / antiphase / slip) are the readout.

``lat_vel_l2`` stays out: strafing suppression is not wobble.

Weight history:

* **v1** (run ``2026-09-14_11-46-53``): the v9-audited walk-task weights
  verbatim (hip_var -100, lin_vel_z/ang_vel_xy -0.5, base_ang_acc -2.5e-4).
  Froze the robot: mean base velocity 0.001 m/s, two legs held in the air,
  vel err 0.63 m/s, height err 14 cm. Those weights worked in the walk task
  only because its positive gait terms subsidized stepping; without that
  subsidy every footfall's torso impulse (priced mainly by base_ang_acc,
  -0.44/s -- 10x any other wobble charge) beats the tracking gradient near
  standstill.
* **v2** (current): same terms cut 10x -- wobble as a bill, not a wall,
  matching the task's energy-pricing philosophy. A tiebreaker between gaits
  that both track, not an argument against walking.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .spidertron_walk_free_env_cfg import SpidertronWalkFreeEnvCfg, WalkFreeRewardsCfg


@configclass
class WalkFreeStableRewardsCfg(WalkFreeRewardsCfg):
    """WalkFree rewards + the torso-stability (anti-wobble) block."""

    hip_height_variance = RewTerm(
        func=mdp.hip_height_variance,
        # hips differing by ~1 cm -> var ~1e-4 -> costs 0.001/step at this weight
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_coxa")},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    base_ang_acc_l2 = RewTerm(
        func=mdp.base_ang_acc_l2,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )


@configclass
class SpidertronWalkFreeStableEnvCfg(SpidertronWalkFreeEnvCfg):
    """WalkFree with wobble penalized. Everything else identical."""

    rewards: WalkFreeStableRewardsCfg = WalkFreeStableRewardsCfg()


@configclass
class SpidertronWalkFreeStableEnvCfg_PLAY(SpidertronWalkFreeStableEnvCfg):
    """Small, deterministic version for watching rollouts."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
