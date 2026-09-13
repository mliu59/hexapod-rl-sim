# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HexapodFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO for the hexapod flat-ground velocity task.

    Network and algorithm settings are held identical to the ANYmal-C rough
    baseline (logs/rsl_rl/anymal_c_rough/2026-07-20_21-07-37/params/agent.yaml)
    so that any difference in training behaviour points at the environment
    rather than the learner. PPO hyperparameters are insensitive to DOF count;
    only the input layer width changes with 18 joints instead of 12.

    The one deliberate change is the rollout length: the hexapod runs at 100 Hz
    instead of 50 Hz, so 24 steps would cover half the wall-clock horizon the
    baseline saw. 48 keeps it at ~0.5 s of experience per env per iteration.
    """

    num_steps_per_env = 48
    max_iterations = 1500
    save_interval = 50
    experiment_name = "hexapod_flat"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class SpidertronStandPPORunnerCfg(HexapodFlatPPORunnerCfg):
    """PPO for the spidertron stand-still task.

    Same learner as the flat-ground velocity baseline -- differences in training
    behaviour should point at the task, not the algorithm. Standing has no
    long-horizon credit assignment, so 300 iterations is generous; watch the
    reward flatten well before that.
    """

    max_iterations = 300
    save_interval = 50
    experiment_name = "spidertron_stand"


@configclass
class SpidertronHeightTrackPPORunnerCfg(SpidertronStandPPORunnerCfg):
    """PPO for the height-setpoint tracking task.

    Harder than standing (the policy must learn transitions between poses),
    so more iterations; everything else inherited. Raised 500 -> 1500 for v2:
    the added contact/planform/jitter constraints conflict with pure height
    speed and need longer to trade off well.
    """

    max_iterations = 1500
    experiment_name = "spidertron_height_track"


@configclass
class SpidertronWalkPPORunnerCfg(SpidertronStandPPORunnerCfg):
    """PPO for the walk task (direction+speed+height, mode-gated rewards).

    The hardest task in the family so far -- gait discovery plus three tracking
    channels plus mode transitions -- hence the longest schedule.
    """

    max_iterations = 2500
    experiment_name = "spidertron_walk"

    def __post_init__(self):
        super().__post_init__()
        # paired with the walk task's 0.35 action scale: keeps the early
        # exploration joint swing (scale x std = 0.245 rad) at or below the
        # proven-survivable 0.25 x 1.0 -- see the v5 spawn-collapse post-mortem
        self.policy.init_noise_std = 0.7


@configclass
class SpidertronMarchPPORunnerCfg(SpidertronWalkPPORunnerCfg):
    """PPO for the march diagnostic task -- short runs for fast A/B iteration."""

    max_iterations = 1000
    experiment_name = "spidertron_march"


@configclass
class SpidertronMarchMaxPPORunnerCfg(SpidertronMarchPPORunnerCfg):
    """PPO for the max-speed march variant."""

    max_iterations = 2000
    experiment_name = "spidertron_march_max"


@configclass
class SpidertronMarchFreePPORunnerCfg(SpidertronMarchMaxPPORunnerCfg):
    """PPO for the free-run (no gait prior) emergent-gait experiment."""

    experiment_name = "spidertron_march_free"
