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
