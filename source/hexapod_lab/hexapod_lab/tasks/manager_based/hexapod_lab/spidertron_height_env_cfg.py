"""Height-tracking task: follow a dynamic base-height setpoint.

The goal-conditioned variant of the stand task. A ``UniformHeightCommand``
(mdp/commands.py) resamples a target hip height per env every 3-6 s; the policy
sees it as one extra observation dim (64 total) and is rewarded for holding the
base there. At deployment any external setpoint source can drive the command
tensor, as long as it stays inside the trained 0.18-0.35 m range and does not
sweep much faster than the step changes trained against.

Range rationale: standing height is 0.2745 m; the leg (0.19 m femur + 0.38 m
tibia reach) has kinematic room well beyond both ends, so the range is set by
stability (high -> support polygon shrinks) and knee torque (low -> crouch
loads the 3.43 N*m STS3250s), not reach.

Two stand-task penalties are deliberately weakened because they fight the task:
``lin_vel_z_l2`` punishes the vertical motion a setpoint change requires, and
``joint_deviation_l1`` anchors to the standing pose while other heights need
other poses.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from . import mdp
from .spidertron_base_env_cfg import ObservationsCfg as BaseObservationsCfg
from .spidertron_base_env_cfg import SpidertronBaseEnvCfg
from .spidertron_stand_env_cfg import StandRewardsCfg

HEIGHT_RANGE = (0.18, 0.35)  # m above ground


@configclass
class CommandsCfg:
    base_height = mdp.UniformHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,
        ranges=mdp.UniformHeightCommandCfg.Ranges(height=HEIGHT_RANGE),
    )


@configclass
class ObservationsCfg(BaseObservationsCfg):
    """Base proprioception plus the height command -- 64 dims."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_height"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class HeightTrackRewardsCfg(StandRewardsCfg):
    """Stand rewards with the height target coming from the command manager."""

    def __post_init__(self):
        # same exp kernel and weight; only the target source changes
        self.base_height_exp.func = mdp.base_height_command_exp
        self.base_height_exp.params = {"command_name": "base_height", "std": 0.03}
        # a setpoint change *requires* vertical velocity and a non-standing pose;
        # keep both terms only as weak regularizers
        self.lin_vel_z_l2.weight = -0.5
        self.joint_deviation_l1.weight = -0.02


@configclass
class SpidertronHeightTrackEnvCfg(SpidertronBaseEnvCfg):
    """Spidertron tracking a piecewise-constant hip-height setpoint."""

    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: HeightTrackRewardsCfg = HeightTrackRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 8 s episodes -> ~2 setpoint changes per episode, so every rollout
        # contains transitions, not just holds


@configclass
class SpidertronHeightTrackEnvCfg_PLAY(SpidertronHeightTrackEnvCfg):
    """Small, deterministic version for watching rollouts."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
