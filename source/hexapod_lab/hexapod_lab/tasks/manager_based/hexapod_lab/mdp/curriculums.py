"""Custom curriculum terms for hexapod tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_duty_metric(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    sensor_name: str,
    foot_body_regex: str = ".*_tibia",
    reduce: str = "min",
) -> float:
    """Log gait structure during training: per-foot duty factor, reduced over feet.

    Not a curriculum -- a live translation probe riding the CurriculumManager's
    logging channel (the one manager hook whose return value lands in
    TensorBoard every iteration; reward-report plots pick it up post-hoc for
    free). ``reduce='min'`` is the tap-dance detector: mean-over-envs of the
    *least-loaded* foot's duty. Near 0 = decorative legs exist; healthy
    six-legged cycling puts it at ~0.4+. ``reduce='mean'`` tracks the overall
    stance/swing balance.
    """
    import re

    contact_sensor = env.scene.sensors[sensor_name]
    ids = [i for i, n in enumerate(contact_sensor.body_names) if re.fullmatch(foot_body_regex, n)]
    ct = contact_sensor.data.last_contact_time[:, ids]
    at = contact_sensor.data.last_air_time[:, ids]
    duty = ct / (ct + at + 1.0e-6)
    reduced = duty.min(dim=1)[0] if reduce == "min" else duty.mean(dim=1)
    return float(reduced.mean())


def tripod_antiphase_metric(env: ManagerBasedRLEnv, env_ids: Sequence[int], sensor_name: str) -> float:
    """Log mean tripod anti-phase |mean_contact(A) - mean_contact(B)| (march v4).

    ~0 for hopping/standing, toward 1 for a clean alternating-tripod gait.
    Same logging channel as foot_duty_metric.
    """
    from .rewards import _TRIPOD_A, _TRIPOD_B

    contact_sensor = env.scene.sensors[sensor_name]
    ids = [i for i, n in enumerate(contact_sensor.body_names) if n.endswith("_tibia")]
    names = [contact_sensor.body_names[i] for i in ids]
    contact = (contact_sensor.data.current_contact_time[:, ids] > 0.0).float()
    a = [k for k, n in enumerate(names) if n[:2] in _TRIPOD_A]
    b = [k for k, n in enumerate(names) if n[:2] in _TRIPOD_B]
    return float((contact[:, a].mean(dim=1) - contact[:, b].mean(dim=1)).abs().mean())


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
