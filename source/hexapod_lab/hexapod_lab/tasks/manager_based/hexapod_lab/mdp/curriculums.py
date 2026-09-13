"""Custom curriculum terms for hexapod tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _foot_ids_cached(env, contact_sensor, foot_body_regex: str) -> list[int]:
    """Resolve foot body indices ONCE. ContactSensor.body_names materializes
    prim paths for the whole batched view (78k strings at 4096 envs) per call;
    doing it per step cost ~7 s/iteration before caching (march v4 post-mortem)."""
    import re

    cache = getattr(env, "_foot_ids_cache", None)
    if cache is None:
        cache = {}
        env._foot_ids_cache = cache
    if foot_body_regex not in cache:
        names = contact_sensor.body_names
        cache[foot_body_regex] = (
            [i for i, n in enumerate(names) if re.fullmatch(foot_body_regex, n)],
            [n for n in names if re.fullmatch(foot_body_regex, n)],
        )
    return cache[foot_body_regex]


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
    contact_sensor = env.scene.sensors[sensor_name]
    ids, _ = _foot_ids_cached(env, contact_sensor, foot_body_regex)
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
    from .rewards import _tripod_indices

    contact_sensor = env.scene.sensors[sensor_name]
    ids, _ = _foot_ids_cached(env, contact_sensor, ".*_tibia")
    contact = (contact_sensor.data.current_contact_time[:, ids] > 0.0).float()
    a, b = _tripod_indices(env, contact_sensor, ids)
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
