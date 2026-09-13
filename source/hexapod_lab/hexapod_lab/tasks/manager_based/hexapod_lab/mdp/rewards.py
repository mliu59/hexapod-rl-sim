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
from isaaclab.utils.math import quat_apply, quat_apply_inverse, wrap_to_pi

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


def base_height_command_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking a *commanded* base height (exp kernel).

    The goal-conditioned counterpart of ``base_height_target_exp``: the target
    comes from the command manager instead of a config constant, so the same
    policy serves any setpoint inside the trained range.

    Assumes flat ground at the env origin height.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, 0]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-torch.square(height - target) / std**2)


def track_forward_vel_hgated_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    height_command_name: str,
    nominal_height: float,
    height_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track the commanded forward (base-x) speed, attenuated away from nominal height.

    Strict-facing walking: only the base-frame forward velocity is commanded
    (the command's own cos-projection already zeroes it while turning in
    place). The attenuation factor ``exp(-((h_cmd - nominal)/height_std)^2)``
    relaxes the speed demand when the commanded height is far from nominal, so
    at crouched/stilted extremes the policy prioritizes holding height over
    hitting speed instead of trading one for the other.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vel_error = torch.square(cmd[:, 0] - asset.data.root_lin_vel_b[:, 0])
    h_cmd = env.command_manager.get_command(height_command_name)[:, 0]
    attenuation = torch.exp(-torch.square((h_cmd - nominal_height) / height_std))
    return torch.exp(-vel_error / std**2) * attenuation


def track_heading_cos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward facing the commanded direction: ``(1 + cos(err)) / 2``.

    Closes the v1 walk loophole (refusing to turn zeroed the speed demand) --
    and, after walk v2, does it with a kernel that has gradient EVERYWHERE on
    the circle. v2 used exp(-err^2/0.5^2), which is ~5e-5 and numerically flat
    at the ~90 deg errors the policy actually had: raising its weight
    multiplied a zero gradient, and heading never trained. The cosine shape
    pays 1 aligned, 0 reversed, with useful slope at every error in between.
    Idle envs (heading pinned) earn it by holding pose, which is desired.
    """
    cmd = env.command_manager.get_command(command_name)
    # cmd[:, 2] is cos(heading error) already
    return 0.5 * (1.0 + cmd[:, 2])


def track_forward_vel_exp(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Track the commanded forward (base-x) speed, exp kernel, no gating.

    The march-task variant of ``track_forward_vel_hgated_exp``: no height
    attenuation (march has no height command) and no other coupling — the
    simplest possible locomotion objective.
    """
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    return torch.exp(-torch.square(cmd[:, 0] - asset.data.root_lin_vel_b[:, 0]) / std**2)


def track_yaw_rate_exp(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Track the heading controller's yaw-rate command (command dim 1, exp kernel)."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    return torch.exp(-torch.square(cmd[:, 1] - asset.data.root_ang_vel_b[:, 2]) / std**2)


def base_lin_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize lateral base velocity (strict facing: strafing is never commanded)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def hip_height_variance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize unequal hip (coxa-link) heights above the ground.

    The torso-flatness measure requested for the walk task: all six leg bases at
    the same height <=> level torso. On flat ground this overlaps
    ``flat_orientation_l2``; it is kept in this form because on rough terrain
    (M3) "hips level with the terrain" and "gravity-level" stop being the same
    thing, and this is the one that generalizes.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    hip_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    return torch.var(hip_z, dim=1)


def foot_slip(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize horizontal foot velocity while in (binary) contact.

    Without this the policy discovers ice-skating: feet report planted while
    sliding, which tracks velocity on flat ground and dies on anything real.
    Both cfgs must name the same bodies (``.*_tibia``) so the index orders match.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    asset: Articulation = env.scene[asset_cfg.name]
    foot_speed_xy = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    return torch.sum(foot_speed_xy * in_contact, dim=1)


def stance_progress(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    slip_tol: float = 0.05,
) -> torch.Tensor:
    """Reward forward progress carried by honestly planted feet.

    ``base forward speed x count(feet in contact AND world-frame foot speed
    below slip_tol)``. This pays for the stride itself rather than its
    airborne byproduct: a sliding foot doesn't count (fails slip_tol), a
    planted robot that isn't moving earns nothing (zero forward speed), and a
    skating gait earns nothing on the feet doing the sliding. The only way to
    collect is the thing we actually want -- translating the body over
    stationary stance feet. Both cfgs must name the same bodies so index
    orders match.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    asset: Articulation = env.scene[asset_cfg.name]
    foot_speed_xy = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    honest = in_contact & (foot_speed_xy < slip_tol)
    forward_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0)
    return forward_speed * torch.sum(honest, dim=1).float()


def torque_saturation(
    env: ManagerBasedRLEnv,
    threshold: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joints operating near their effort limit (anti-dithering).

    ``mean over joints of max(0, |tau|/limit - threshold)`` — charges only the
    top decile of effort, the signature of the walk-v8 failure: bang-bang
    position commands beyond servo bandwidth leave joints pinned at the torque
    limit (RR coxa: 97.8% of steps >90% limit) while the servo low-passes the
    thrash. ``dof_torques_l2`` cannot see this (torque magnitude is capped at
    the limit); this term prices time-at-saturation directly. Also the
    hardware-lethality metric: a real STS servo held at stall overheats.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    frac = asset.data.applied_torque.abs() / asset.data.joint_effort_limits.clamp(min=1e-6)
    return torch.mean(torch.clamp(frac - threshold, min=0.0), dim=1)


def feet_airborne_too_long(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    max_air_time: float,
) -> torch.Tensor:
    """Count feet held airborne beyond ``max_air_time`` while commanded moving.

    Closes the tripod-hop loophole from walk v8: `feet_off_ground` is gated to
    idle envs and `feet_air_time`'s floor only prices *completed* swings at
    touchdown — a leg that never lands triggers neither. This charges, every
    step, for any foot whose CURRENT air time exceeds the cap, so carrying
    curled legs while moving accrues cost continuously. Cap well above the
    0.4 s target swing so honest strides never pay.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    too_long = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > max_air_time
    active = env.command_manager.get_term(command_name).is_active
    return torch.sum(too_long, dim=1).float() * active


def foot_duty_deviation(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    target: float = 0.6,
    tol: float = 0.15,
) -> torch.Tensor:
    """Penalize per-foot contact duty factor outside [target-tol, target+tol] while moving.

    The load-coupling term motivated by the v9 tap-dance exploit
    (docs/SPIDERTRON_TASKS.md): every earlier gait term treated feet
    independently, so the policy scuttled on two legs (duty ~55%) while four
    decorative legs farmed the air-time reward at ~2% duty with periodic taps.
    Duty factor -- contact time as a fraction of the last completed
    stance+swing cycle -- makes both extremes illegal per foot: a decorative
    leg (~0.02) and an always-planted drag leg (~1.0) both pay, and only
    genuine load-sharing cycling (~0.45-0.75 at the defaults) is free.
    Combined with the air-time floor (swings must also be long enough), the
    cheapest legal strategy is an actual gait.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    ct = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    at = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    duty = ct / (ct + at + 1.0e-6)
    dev = torch.clamp((duty - target).abs() - tol, min=0.0)
    return torch.sum(dev, dim=1) * env.command_manager.get_term(command_name).is_active


def _tripod_indices(env, contact_sensor: ContactSensor, body_ids) -> tuple[list[int], list[int]]:
    """The two alternating-tripod leg sets, derived from mount GEOMETRY.

    Sort the legs by their coxa mount angle around the body and take every
    other one: the unique 2-coloring of a leg ring with no adjacent legs in
    the same set (up to A/B swap). This grouping is rotation-invariant and
    direction-agnostic -- both support triangles contain the COM for travel
    in any direction -- so it prescribes no body orientation, and the same
    derivation works for any radially-legged morphology. (For this robot it
    resolves to {LF, LR, RM} vs {LM, RR, RF}.)

    CACHED on the env: ContactSensor.body_names materializes prim paths for
    every body in the batched view (78k strings at 4096 envs) on EVERY call
    -- calling it per step cost ~1 s/step and collapsed throughput 20x
    (march v4 post-mortem). Resolve once.
    """
    cache = getattr(env, "_tripod_indices_cache", None)
    if cache is None:
        names = [contact_sensor.body_names[i] for i in body_ids]
        order = sorted(range(len(names)), key=lambda k: _LEG_MOUNT_ANGLES[names[k][:2]])
        cache = (order[0::2], order[1::2])
        env._tripod_indices_cache = cache
    return cache


def tripod_antiphase(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Reward anti-phase loading of the two tripod sets while moving.

    ``|mean_contact(tripod A) - mean_contact(tripod B)|`` in [0, 1]: maximal
    when one tripod is planted and the other swings (the alternating-tripod
    gait), and exactly ZERO for the failure modes seen on flat ground -- a
    synchronized hop (both tripods airborne) and static standing (both
    planted) both score nothing. Dense every step; combined with the air-time
    and duty terms, alternation over time is the only way to collect it.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = (contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float()
    a, b = _tripod_indices(env, contact_sensor, sensor_cfg.body_ids)
    diff = (contact[:, a].mean(dim=1) - contact[:, b].mean(dim=1)).abs()
    return diff * env.command_manager.get_term(command_name).is_active


def too_many_feet_airborne(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    max_airborne: int = 3,
) -> torch.Tensor:
    """Penalize more than ``max_airborne`` feet off the ground while moving.

    The direct anti-hop constraint: a proper alternating-tripod gait never
    needs more than 3 feet in flight, while a hop lifts 4-6 at once. Charged
    per extra airborne foot per step, so a full flight phase (6 airborne)
    costs 3x the penalty weight continuously. On rough terrain hops would be
    selected against by falls; on flat ground this term does that job.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    airborne = (~in_contact).sum(dim=1).float()
    excess = torch.clamp(airborne - float(max_airborne), min=0.0)
    return excess * env.command_manager.get_term(command_name).is_active


def feet_off_ground_idle(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """`feet_off_ground`, gated to idle envs (commanded neither moving nor turning)."""
    idle = ~env.command_manager.get_term(command_name).is_active
    return feet_off_ground(env, sensor_cfg) * idle


def feet_air_time_target_active(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_time: float,
    min_air_time: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated on the motion command's is_active flag, with a
    penalty floor for micro-hops.

    Shape: ``clamp(air_time - min_air_time, max=target - min_air_time)`` at
    each touchdown. The earlier symmetric peak-at-target shape still paid tiny
    hops positively, and hops touch down 3-4x as often as real swings -- per
    unit time, shuffling earned nearly as much as stepping, which is exactly
    the gait the walk-v1/v2 rollouts showed. With the floor, a swing shorter
    than ``min_air_time`` COSTS reward, and its high touchdown frequency
    multiplies the cost; a full swing pays the cap.

    Gate: the command term's moving-or-turning boolean (turn-in-place needs
    stepping too, and this task's command dim 0 is zero while turning).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    per_foot = torch.clamp(last_air_time - min_air_time, max=target_time - min_air_time) * first_contact
    return torch.sum(per_foot, dim=1) * env.command_manager.get_term(command_name).is_active


def air_time_variance_active(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str
) -> torch.Tensor:
    """`air_time_variance`, gated on the motion command's is_active flag (stale
    swing statistics would otherwise be penalized while standing)."""
    return air_time_variance(env, sensor_cfg) * env.command_manager.get_term(command_name).is_active


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


def _foot_positions_w(asset: Articulation, body_ids, tip_offset: float = TIBIA_TIP_OFFSET) -> torch.Tensor:
    """World positions of the tibia-tip contact spheres. Shape is (num_envs, num_feet, 3)."""
    pos_w = asset.data.body_pos_w[:, body_ids]
    quat_w = asset.data.body_quat_w[:, body_ids]
    offset = torch.zeros_like(pos_w)
    offset[..., 0] = tip_offset
    return pos_w + quat_apply(quat_w, offset)


def feet_off_ground(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Count of feet not in contact with the ground (binary contact detection).

    For stationary tasks (stand, height-track) every foot should be planted the
    whole time; a policy that balances on 3-4 legs and waves the rest satisfies
    the base-height reward just as well.

    Contact is a per-foot BOOLEAN from the sensor's contact-time state machine
    (``current_contact_time > 0``, driven by the sensor cfg's single
    ``force_threshold``) -- task logic never sees force magnitudes, matching a
    real robot's contact-switch feet.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    return torch.sum(~in_contact, dim=1).float()


def base_yaw_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize yaw angular velocity (for tasks that should hold heading).

    Note the policy cannot observe absolute yaw (projected gravity is
    yaw-blind and there is no compass term), so an absolute-heading penalty
    would be unlearnable; damping the yaw *rate* is the observable equivalent
    and pins whatever heading the episode started with.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def base_ang_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base angular acceleration (orientation jitter).

    ``ang_vel_xy_l2`` penalizes sustained tilt rates but is nearly blind to
    high-frequency oscillation, whose velocity amplitude is small while its
    acceleration is large. This is the base-frame analogue of ``joint_acc_l2``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.body_ang_acc_w[:, asset_cfg.body_ids].squeeze(1)), dim=-1)


# Leg mount directions (rad, base frame, from the URDF coxa joint origins):
# legs are radial on a perfect hexagon, so each foot's nominal planform
# position is radius * (cos, sin) of its mount angle.
_LEG_MOUNT_ANGLES = {
    "LF": 0.523599,
    "LM": 1.570796,
    "LR": 2.617994,
    "RR": 3.665191,
    "RM": 4.712389,
    "RF": 5.759587,
}


def feet_position_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    radius: float,
    tip_offset: float,
) -> torch.Tensor:
    """Penalize feet drifting from their nominal planform positions.

    Nominal is equal hexagonal spacing at ``radius`` from the base origin
    (evaluated in the base frame, so it turns with the robot). Constrains both
    the radial drift (feet sliding out or in) and the angular spread the other
    terms leave free.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cache_key = "_nominal_foot_xy"
    nominal = getattr(env, cache_key, None)
    if nominal is None:
        names = [asset.body_names[i] for i in asset_cfg.body_ids]
        angles = torch.tensor([_LEG_MOUNT_ANGLES[name[:2]] for name in names], device=env.device)
        nominal = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        setattr(env, cache_key, nominal)
    foot_pos_w = _foot_positions_w(asset, asset_cfg.body_ids, tip_offset)
    rel_w = foot_pos_w - asset.data.root_pos_w.unsqueeze(1)
    quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, rel_w.shape[1], -1)
    rel_b = quat_apply_inverse(quat, rel_w)
    return torch.sum(torch.square(rel_b[..., :2] - nominal), dim=(1, 2))


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
    tip_offset: float = TIBIA_TIP_OFFSET,
) -> torch.Tensor:
    """Penalize swing feet that do not reach ``target_height`` above the ground.

    Weighted by each foot's horizontal speed, so a planted foot contributes nothing and only
    feet that are actually swinging are asked to lift. Without this the policy discovers that
    scuffing the feet along the ground tracks velocity perfectly well on flat terrain -- and
    then falls over the first obstacle it meets in M3.

    Assumes flat ground at the env origin height.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = _foot_positions_w(asset, asset_cfg.body_ids, tip_offset)
    foot_height = foot_pos_w[..., 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    # body-origin velocity is a good enough proxy for the tip's horizontal speed here
    foot_speed_xy = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    return torch.sum(torch.square(foot_height - target_height) * foot_speed_xy, dim=1)
