"""Custom command terms for hexapod tasks.

Isaac Lab's CommandManager is how a task exposes an external setpoint to the
policy: a command term owns a ``(num_envs, dim)`` tensor, resamples it per-env
on a timer, and the policy sees it through an ``mdp.generated_commands``
observation term while rewards read it via ``env.command_manager.get_command``.
The policy trained this way is goal-conditioned -- at deployment the tensor can
be driven by any external function whose values stay inside the trained range
(and whose rate of change resembles the resampling the policy saw).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformHeightCommand(CommandTerm):
    """Piecewise-constant base-height setpoint, sampled uniformly from a range.

    Command is shape (num_envs, 1): the desired base height above the env
    origin, in meters. Logged metric ``error_height`` is the time-averaged
    absolute tracking error, the height analogue of ``error_vel_xy``.
    """

    cfg: UniformHeightCommandCfg

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"UniformHeightCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tHeight range: {self.cfg.ranges.height} m\n"
            f"\tResampling time range: {self.cfg.resampling_time_range} s"
        )

    @property
    def command(self) -> torch.Tensor:
        """Desired base height above the env origin. Shape is (num_envs, 1)."""
        return self.height_command

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        height = self.robot.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        self.metrics["error_height"] += torch.abs(self.height_command[:, 0] - height) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)

    def _update_command(self):
        pass


class DirectionSpeedCommand(CommandTerm):
    """Walk-along-a-direction command with strict facing (heading mode).

    Samples a world-frame target heading and a forward speed per env. Every step
    a proportional controller converts heading error into a yaw-rate command,
    and the forward-speed target is the sampled speed projected onto the facing
    error (``speed * max(cos(err), 0)``): a robot facing the wrong way is asked
    to turn, not translate, so turn-in-place emerges when the target is behind
    it. No lateral velocity is ever commanded (strict facing, no strafing).

    Command tensor (num_envs, 4), all base-frame observable without a compass:
    ``[forward speed target, yaw-rate command, cos(heading err), sin(heading err)]``.

    A ``rel_standing_envs`` fraction is commanded idle: zero speed, heading
    pinned to the robot's heading at resample time (so it holds pose rather
    than turning). ``is_active`` exposes the moving-or-turning boolean that
    gait/stand reward terms gate on.
    """

    cfg: DirectionSpeedCommandCfg

    def __init__(self, cfg: DirectionSpeedCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.speed = torch.zeros(self.num_envs, device=self.device)
        self.motion_command = torch.zeros(self.num_envs, 4, device=self.device)
        self.is_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["error_vel_fwd"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"DirectionSpeedCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tSpeed range: {self.cfg.ranges.speed} m/s\n"
            f"\tMax yaw rate: {self.cfg.max_yaw_rate} rad/s (stiffness {self.cfg.heading_control_stiffness})\n"
            f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range} s"
        )

    @property
    def command(self) -> torch.Tensor:
        """[v_fwd target, yaw-rate cmd, cos(heading err), sin(heading err)]. Shape is (num_envs, 4)."""
        return self.motion_command

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        err = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
        self.metrics["error_vel_fwd"] += (
            torch.abs(self.motion_command[:, 0] - self.robot.data.root_lin_vel_b[:, 0]) / max_command_step
        )
        self.metrics["error_heading"] += torch.abs(err) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
        self.speed[env_ids] = r.uniform_(*self.cfg.ranges.speed)
        standing = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        standing_ids = torch.as_tensor(env_ids, device=self.device)[standing]
        self.speed[standing_ids] = 0.0
        # idle means idle: pin the target heading to the current one so the
        # heading controller holds pose instead of commanding a turn
        self.heading_target[standing_ids] = self.robot.data.heading_w[standing_ids]

    def _update_command(self):
        err = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
        yaw_rate = torch.clip(
            self.cfg.heading_control_stiffness * err, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate
        )
        # softened cos-projection: full speed when facing, zero beyond 120 deg
        # misalignment (not 90). The v1 hard clamp let a robot that refused to
        # turn zero its own speed demand and collect the tracking reward for
        # free -- see docs/SPIDERTRON_TASKS.md iteration 3.
        v_fwd = self.speed * torch.clamp((torch.cos(err) + 0.5) / 1.5, min=0.0, max=1.0)
        self.motion_command[:, 0] = v_fwd
        self.motion_command[:, 1] = yaw_rate
        self.motion_command[:, 2] = torch.cos(err)
        self.motion_command[:, 3] = torch.sin(err)
        self.is_active = (self.speed > 0.02) | (yaw_rate.abs() > 0.1)


@configclass
class DirectionSpeedCommandCfg(CommandTermCfg):
    """Configuration for the direction+speed (heading mode) command generator."""

    class_type: type = DirectionSpeedCommand

    asset_name: str = MISSING
    """Name of the articulation the command drives."""

    heading_control_stiffness: float = 0.5
    """Proportional gain mapping heading error (rad) to yaw-rate command (rad/s)."""

    max_yaw_rate: float = 0.8
    """Yaw-rate command clip (rad/s); doubles as the turn-in-place speed."""

    rel_standing_envs: float = 0.1
    """Fraction of envs commanded idle (zero speed, heading pinned) at resample."""

    @configclass
    class Ranges:
        heading: tuple[float, float] = MISSING
        """World-frame target heading range (rad)."""

        speed: tuple[float, float] = MISSING
        """Forward speed range (m/s, non-negative)."""

    ranges: Ranges = MISSING


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for the uniform height command generator."""

    class_type: type = UniformHeightCommand

    asset_name: str = MISSING
    """Name of the articulation whose base height is commanded."""

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING
        """Setpoint range (min, max) in meters above the env origin."""

    ranges: Ranges = MISSING
