"""Custom observation terms for hexapod tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Binary per-foot contact states as floats. Shape (num_envs, num_feet).

    The gait rewards all key on contact events; without this the policy could
    only infer contact from joint-velocity transients — graded on something it
    couldn't see. Matches what real foot micro-switches would provide (the
    sensor's single force_threshold is the one place the boolean is derived).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    return (contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float()


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
