"""Custom curriculum terms for hexapod tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_speed_ramp(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    start_speed: float,
    end_speed: float,
    ramp_steps: int,
) -> float:
    """Linearly widen the motion command's speed range over training.

    The walk-v8 discovery fix: at low commanded speeds, taking a real step
    barely disturbs velocity/height tracking, so the barrier between the
    shuffle local optimum and a true gait is thin -- the policy learns to step
    at ~0.1 m/s and carries the gait upward as the range widens. Three rounds
    of penalty pricing (v4-v7) moved the equilibrium but never produced the
    coordinated leap; this changes what is discoverable instead of what it
    costs.

    Mutates the command term's cfg speed range (read at every resample).
    Returns the current top speed, logged under ``Curriculum/``.
    """
    frac = min(env.common_step_counter / ramp_steps, 1.0)
    top_speed = start_speed + (end_speed - start_speed) * frac
    env.command_manager.get_term(command_name).cfg.ranges.speed = (0.0, top_speed)
    return top_speed
