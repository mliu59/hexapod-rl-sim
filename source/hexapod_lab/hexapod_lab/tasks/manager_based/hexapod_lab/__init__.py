# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Hexapod-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_flat_env_cfg:HexapodFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerCfg",
    },
)

gym.register(
    id="Hexapod-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_flat_env_cfg:HexapodFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_stand_env_cfg:SpidertronStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronStandPPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-HeightTrack-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_height_env_cfg:SpidertronHeightTrackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronHeightTrackPPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-HeightTrack-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_height_env_cfg:SpidertronHeightTrackEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronHeightTrackPPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_walk_env_cfg:SpidertronWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronWalkPPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-Walk-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_walk_env_cfg:SpidertronWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronWalkPPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-March-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_march_env_cfg:SpidertronMarchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronMarchPPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-MarchMax-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_march_env_cfg:SpidertronMarchMaxEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronMarchMaxPPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-MarchMax-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_march_env_cfg:SpidertronMarchMaxEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronMarchMaxPPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-MarchFree-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_march_env_cfg:SpidertronMarchFreeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronMarchFreePPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-MarchFree-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_march_env_cfg:SpidertronMarchFreeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronMarchFreePPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-March-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_march_env_cfg:SpidertronMarchEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronMarchPPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-WalkFree-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_walk_free_env_cfg:SpidertronWalkFreeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronWalkFreePPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-WalkFree-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_walk_free_env_cfg:SpidertronWalkFreeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronWalkFreePPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-WalkFreeStable-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_walk_free_stable_env_cfg:SpidertronWalkFreeStableEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronWalkFreeStablePPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-WalkFreeStable-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_walk_free_stable_env_cfg:SpidertronWalkFreeStableEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronWalkFreeStablePPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-WalkFreeStable2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_walk_free_stable2_env_cfg:SpidertronWalkFreeStable2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronWalkFreeStable2PPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-WalkFreeStable2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_walk_free_stable2_env_cfg:SpidertronWalkFreeStable2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronWalkFreeStable2PPORunnerCfg",
    },
)

gym.register(
    id="Spidertron-Stand-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spidertron_stand_env_cfg:SpidertronStandEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpidertronStandPPORunnerCfg",
    },
)