"""Custom observation terms for hexapod tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def height_setpoint_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Signed error between the commanded and current base height (m). Shape (num_envs, 1).

    Gives the policy the tracking error directly instead of making it infer its
    own height from leg geometry -- the servo-loop input, positive when the base
    must rise. Assumes flat ground at the env origin height.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, :1]
    height = asset.data.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]
    return target - height
