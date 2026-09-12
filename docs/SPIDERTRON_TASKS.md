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
