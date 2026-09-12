# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# The hexapod has no separate foot link: the contact point is a sphere at the
# far end of the tibia cylinder, 0.14 m along the tibia's local +x axis.
# Keep in sync with tibia_length in scripts/generate_hexapod_urdf.py.
TIBIA_TIP_OFFSET = 0.14


def base_height_target_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the base sitting at ``target_height`` above the env origin (exp kernel).

    The positive counterpart of ``mdp.base_height_l2``: a task whose *objective* is
    holding a height needs a term that pays for success, not one that only punishes
    deviation -- an all-penalty reward gives the policy nothing to climb and makes
    the reward report unreadable (every term negative, "best" run least-punished).
    ``std`` sets the tolerance: reward is ~0.37 of max at ``|h - target| = std``.

    Assumes flat ground at the env origin height.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-torch.square(height - target_height) / std**2)


def base_lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize horizontal base velocity (for stand-still tasks).

    The core mdp only offers ``lin_vel_z_l2``; a velocity-tracking task keeps the
    xy plane free for the command, but a stationary task wants it damped too.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def _foot_positions_w(asset: Articulation, body_ids) -> torch.Tensor:
    """World positions of the tibia-tip contact spheres. Shape is (num_envs, num_feet, 3)."""
    pos_w = asset.data.body_pos_w[:, body_ids]
    quat_w = asset.data.body_quat_w[:, body_ids]
    offset = torch.zeros_like(pos_w)
    offset[..., 0] = TIBIA_TIP_OFFSET
    return pos_w + quat_apply(quat_w, offset)


def feet_air_time_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_time: float,
) -> torch.Tensor:
    """Reward per-leg swing durations close to ``target_time``, evaluated at touchdown.

    The quadruped ``feet_air_time`` term rewards air time in excess of a threshold, which
    encodes a trot prior: longer swings are always better. A hexapod carrying its weight on
    a tripod cannot swing that long -- XM430-class servos cap joint speed at 4.8 rad/s -- so
    an unbounded term just asks for something the actuators cannot deliver. This one peaks
    when a leg's swing lasts ``target_time`` and falls off symmetrically either side, which
    regularizes the gait period instead of stretching it.

    Zero while the robot is commanded to stand still.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # peaks at target_time, decreasing linearly in |air_time - target_time|
    per_foot = (target_time - (last_air_time - target_time).abs()) * first_contact
    reward = torch.sum(per_foot, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return reward


def air_time_variance(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize uneven use of the six legs (per-leg air-time regularization).

    Nothing in the velocity-tracking reward requires all legs to contribute. With six legs
    there is enough redundancy to hit the commanded velocity while dragging one or two, which
    looks fine in the reward curve and terrible in the rollout. Penalizing the spread of swing
    and stance durations across legs pushes toward an even, repeatable gait.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(last_air_time, dim=1) + torch.var(last_contact_time, dim=1)


def foot_clearance_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
) -> torch.Tensor:
    """Penalize swing feet that do not reach ``target_height`` above the ground.

    Weighted by each foot's horizontal speed, so a planted foot contributes nothing and only
    feet that are actually swinging are asked to lift. Without this the policy discovers that
    scuffing the feet along the ground tracks velocity perfectly well on flat terrain -- and
    then falls over the first obstacle it meets in M3.

    Assumes flat ground at the env origin height.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = _foot_positions_w(asset, asset_cfg.body_ids)
    foot_height = foot_pos_w[..., 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    # body-origin velocity is a good enough proxy for the tip's horizontal speed here
    foot_speed_xy = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    return torch.sum(torch.square(foot_height - target_height) * foot_speed_xy, dim=1)
