# Spidertron task family — pipeline, tasks, results

Status log for RL tasks on the CAD-derived spidertron robot
([robots/spidertron.py](../source/hexapod_lab/hexapod_lab/robots/spidertron.py),
Feetech STS servos, 0.2745 m standing hip height). Each task follows the
pipeline established in
[spidertron_base_env_cfg.py](../source/hexapod_lab/hexapod_lab/tasks/manager_based/hexapod_lab/spidertron_base_env_cfg.py):
subclass the base config, add a rewards configclass (custom terms in
`mdp/rewards.py`, custom commands in `mdp/commands.py`), a runner config, and a
`gym.register` entry. All runs: 4096 envs, PPO (rsl-rl), 100 Hz control,
RTX 4070 Ti, launched via `scripts/rsl_rl/train.py --task <id> --headless`
under the observability scaffold.

## Iteration 1 — `Spidertron-Stand-v0` (2026-09-12)

Trivial first task: hold the base at the nominal 0.2745 m hip height. One
positive term (`base_height_target_exp`, exp kernel, std 3 cm) plus stillness
and contact penalties. No commands; 63-dim proprioceptive obs.

**Results** (300 iters, 9.2 min, ~114k steps/s, 5.80 GB VRAM peak):
`base_height_exp` 1.998 / 2.0 max, all penalties < 0.03, zero falls, episode
length pinned at timeout. Visually verified.

- Rollout: [stand_spidertron.mp4](stand_spidertron.mp4)
- Curves: [stand_spidertron_training.png](stand_spidertron_training.png)
- Run dir: `logs/rsl_rl/spidertron_stand/2026-09-12_13-13-18/` (local)

## Iteration 2 — `Spidertron-HeightTrack-v0` (2026-09-12)

Goal-conditioned variant: follow a **dynamic hip-height setpoint**. A custom
`UniformHeightCommand`
([mdp/commands.py](../source/hexapod_lab/hexapod_lab/tasks/manager_based/hexapod_lab/mdp/commands.py))
resamples a per-env target uniformly in **0.18–0.35 m** every 3–6 s; the policy
observes it as a 64th obs dim and `base_height_command_exp` rewards tracking
it. Two stand-task penalties are weakened because they fight the objective:
`lin_vel_z_l2` −2.0 → −0.5 (setpoint changes require vertical velocity) and
`joint_deviation_l1` −0.1 → −0.02 (other heights need other poses). Range set
by stability/knee torque, not kinematic reach (legs: 0.19 m femur + 0.38 m
tibia reach).

**Training** (500 iters, 15.8 min, 5.80 GB VRAM peak):
`base_height_exp` 1.95 / 2.0 (the gap is transition time after setpoint
steps), time-averaged `error_height` 4.2 mm including transients, zero falls.

**Staircase evaluation** — `scripts/demo_height_staircase.py` drives the
command tensor with a scripted external profile (the deployment pattern: any
generator can write the tensor; training only fixes the valid range and
step-like rate). Setpoints 0.2745 → 0.20 → 0.30 → 0.35 → 0.25 → 0.18 → 0.2745 m,
2.5 s holds, single env:

| Target (m) | Settle to ±1 cm | Steady-state error |
|-----------:|----------------:|-------------------:|
| 0.20       | 0.22 s          | 0.9 mm             |
| 0.30       | 0.17 s          | 1.0 mm             |
| 0.35       | 0.14 s          | 1.7 mm             |
| 0.25       | 0.12 s          | 1.2 mm             |
| 0.18       | 0.26 s          | 2.0 mm             |
| 0.2745     | 0.15 s          | 1.9 mm             |

Slowest settles are the deep-crouch targets, consistent with knee-torque cost.

- Rollout (32 envs, random setpoints): [height_track_spidertron.mp4](height_track_spidertron.mp4)
- Staircase demo (1 env, scripted): [height_staircase_spidertron.mp4](height_staircase_spidertron.mp4)
- Curves: [height_track_spidertron_training.png](height_track_spidertron_training.png)
- **Final policy**: [policies/spidertron_height_track/](policies/spidertron_height_track/) —
  `model_499.pt` (rsl-rl checkpoint, works with `--checkpoint` on
  `play.py`/`demo_height_staircase.py`), `policy.pt` (TorchScript) and
  `policy.onnx` (deployment exports, obs layout below)
- Run dir: `logs/rsl_rl/spidertron_height_track/2026-09-12_13-39-40/` (local)

**Obs layout for the exported policy** (64 dims, base frame, positions/vels
relative to default pose): base_lin_vel(3), base_ang_vel(3),
projected_gravity(3), joint_pos_rel(18), joint_vel_rel(18), last_action(18),
height_command(1). Action: 18 joint-position offsets, scale 0.25 rad about the
standing pose.

**v2 addendum** (2026-09-12, run `2026-09-12_15-03-35`): rollout review of v1
prompted four added constraints — `feet_off_ground` (all six feet planted,
binary contact), `yaw_rate_l2` (hold heading), `base_ang_acc_l2` (orientation
jitter), and a foot-planform hexagon pin — plus a `height_setpoint_error`
obs (65 dims). Constraints all converged near zero cost, but the planform pin
(largest converged penalty, −0.33) cost up to 26 mm of height accuracy at the
range extremes and was **removed as cosmetic**; the other three stayed. Foot
contact was also simplified to **binary detection** (single `force_threshold`
on the sensor cfg; task logic never reads force magnitudes).

## Iteration 3 — `Spidertron-Walk-v0` v1 (2026-09-12): FAILED on yaw, diagnosed

Walk task: direction+speed command (heading mode: P-controller turns heading
error into a yaw-rate command; forward-speed target cos-projected so facing
away commands turning, not translation; strict facing, no strafing) + the
height setpoint + mode-gated rewards (all-feet-down when idle, air-time gait
shaping when moving). 69-dim obs, 21 reward terms, 12 s episodes,
`DirectionSpeedCommand` in `mdp/commands.py`. 2500 iters / 1 h 46 min, run
`logs/rsl_rl/spidertron_walk/2026-09-12_16-16-31/` (local, incl. per-200-iter
progress videos in `videos/demo/`).

**Outcome**: forward-speed tracking 1.41/2.0 (err ~0.07 m/s), height 1.59/2.0
(err 2.7 cm), all stability constraints held, zero falls — but
**`track_yaw_rate` 0.005/1.0 and ~95° mean heading error: the robot never
faces the target**, and no real gait emerged (`feet_air_time` 0.04/4.0).

**Diagnosis — reward loophole, not undertraining**: the cos-projection
`v_fwd_target = speed * max(cos(err), 0)` zeroes the speed demand for a robot
that refuses to turn, so ignoring the heading earns the forward-vel reward for
free (tracking a zero target while standing), loses only the weight-1.0 yaw
term, and avoids every stepping cost. PPO found and kept that optimum; the
curve was still "improving" while the task was failing — a reminder that the
mean-reward curve cannot distinguish tracking from loophole exploitation.

**v2 fixes**: direct heading-alignment reward `track_heading_exp` (+1.5, exp
kernel std 0.5) so facing pays on its own channel; `track_yaw_rate` weight
1.0 → 2.0; projection softened to reach zero at 120° misalignment instead of
90° (`clamp((cos+0.5)/1.5, 0, 1)`).

**v2 outcome** (run `2026-09-12_18-17-04`, stopped at iteration ~900 by
design review): heading still did not train — `track_heading` flat at
0.22–0.26 from iteration 200 on, mean heading error ~85–90°, yaw-rate reward
flat at 0.05 — while total reward climbed on the other channels. **Second
diagnosis: kernel reach, not weights.** Both v2 alignment rewards are exp
kernels that are numerically flat at the errors the policy actually had
(exp(−(1.57/0.5)²) ≈ 5e-5 at 90° heading error; exp(−4) ≈ 0.02 for a fully
untracked yaw command). Raising the weight multiplied a zero gradient; PPO
had no slope to climb toward turning.

**v3 revision** (in config): `track_heading_cos` = (1+cos(err))/2 — gradient
everywhere on the circle, 1 aligned / 0 reversed; `track_yaw_rate` std widened
0.4 → 0.8 (full command range) so the slope is usable even from zero turning.
Lesson for the file: when a tracking term stays flat while total reward
climbs, check the kernel's value AT the current operating error before
touching weights — a reward the policy never samples the slope of does not
exist.

**v3/v4 gait shaping** (same relaunch): the v1/v2 gait was a micro-hop
shuffle, traced to the symmetric air-time reward paying tiny hops positively
per touchdown (and hops touch down 3-4x as often as swings, so
reward-per-second was indifferent). Reshaped to
`clamp(air − MIN_AIR, max = TARGET − MIN_AIR)`: sub-floor hops now COST at
every touchdown. v4 pushed for longer strides on request: TARGET 0.3 → 0.4 s,
MIN_AIR 0.1 → 0.15 s (~12 cm stance travel per cycle at 0.3 m/s, within coxa
range).

**v4 outcome** (run `2026-09-12_19-06-25`, stopped at ~840): the kernel fixes
worked decisively — `track_heading` 1.28/1.5 and `track_yaw_rate` 1.69/2.0 by
iteration 600, vs flat-zero through v2 — but gait stalled in a
suppressed-shuffle equilibrium: `feet_air_time` flat at ≈ −0.02 for 600
iterations while everything else converged. **v5 revision**: action scale
0.25 → 0.35 rad (walk task only — hypothesis: 0.4 s swings need more
joint-angle room per action than 0.25 rad expresses) and `feet_air_time`
weight 4.0 → 6.0.

**v5 outcome — instant collapse** (run `2026-09-12_19-52-46`, stopped at
300): every env fell on its chassis at spawn (24-step episodes, 100%
base_contact terminations, no recovery). Cause: action scale × init
exploration std (1.0) is the early-training joint swing; 0.35 rad exceeds
what the knees survive, the same saturation mode as the spawn-height lesson.
**v6**: scale reverted to 0.25, keeping only the `feet_air_time` weight 6.0 —
a clean single-variable test of the weight lever. If stride room is revisited,
widen scale only together with lower init noise (keep the product ≤ 0.25).
