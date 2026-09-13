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

**v6 outcome — weight lever insufficient, mechanism identified** (run
`2026-09-12_20-07-46`, stopped at ~800): healthy training (full episodes,
heading/yaw converging on the v4 trajectory), but `feet_air_time` flat at
≈ −0.009 through iteration 800. The tell was in the other terms: forward
tracking healthy (1.39) with `foot_slip` −0.13 and `air_time_variance` −0.29 —
the policy moves by **sliding nominally-planted feet** (skating is cheaper
than stepping at slip −0.5), while the variance penalty **taxes stepping
exploration** (irregular early steps cost immediately, regular gait pays only
later). **v7**: foot_slip −0.5 → −2.0 (skating must cost more than stepping),
air_time_variance −0.25 → −0.1 (lower the exploration tax), and stride room
done safely — action scale 0.35 paired with init_noise_std 0.7 (exploration
product 0.245 ≤ the proven 0.25).

**v7 outcome — pricing has hit its limit** (run `2026-09-12_20-48-27`,
stopped at ~970): `feet_air_time` went positive for the first time ever
(+0.005 at iteration 500) then regressed to the familiar −0.009; actual foot
sliding halved but the gait never assembled. Conclusion after three pricing
rounds (v4/v6/v7): cost tuning moves the shuffle equilibrium but cannot
produce the coordinated leap to a stride cycle — crossing it requires
transiently degrading tracking, and PPO's local exploration never samples a
full coherent stride. This is a DISCOVERY problem, not a pricing problem.

**v8 outcome — curriculum falsified; root cause found via joint diagnostics**
(run `2026-09-12_22-03-50`, full 2500 iters / 1 h 51 min): tracking excellent
(fwd err 7.6 cm/s, height err 2.3 cm, heading/yaw strong, zero falls), but
`feet_air_time` never inflected at ANY ramp speed (−0.024 at end) and
`stance_progress` stayed marginal (+0.0025). The decisive evidence came from
the new `--diagnostics` mode on `demo_walk_rollout.py` (per-joint
target-vs-measured + torque saturation, prompted by a leg that looked stuck
in the iter-600 rollout): the policy commands full-range bang-bang
oscillations far beyond servo bandwidth — RR coxa mean tracking error
1.06 rad with **97.8% of steps above 90% of the effort limit**. The apparent
gait is saturation-averaged dithering: the servos low-pass the thrash, and
the visible pose is its time-average. Not intentional placement, not a
mechanical jam — a hardware-lethal control style (a real STS servo at ~stall
would overheat in minutes), and the real reason no pricing or curriculum
ever produced a stride: the policy never learned smooth trajectories to
build one from.

Visually (final rollout, `walk_rollout_hud_final.mp4`), the converged
locomotion reads as **bouncing/jumping on ~3 legs**: several legs (RR
foremost) are held curled in the dithering tuck, and the remainder
pogo the body forward with the dither providing the excitation. This is
consistent with every metric at once — it tracks velocity (the hops
average to the commanded speed), it keeps `feet_air_time` slightly
negative (hop flights are shorter than the 0.15 s floor), it defeats
`feet_off_ground` (that term is gated to idle envs only, and hop-contacts
re-trigger constantly while moving), and it partially evades `foot_slip`
(airborne feet can't slip). A three-legged pogo is also exactly the
morphology-abuse a statically-stable hexapod can afford that a biped or
quadruped could not. **v9 direction**: force smoothness first — action_rate_l2
raised an order of magnitude and/or a dedicated torque-saturation penalty
(the diagnostic's own metric as a term) — then re-test gait discovery.

**v9 (in config)** — smoothness-first package plus a reward/obs audit, the
largest delta of the arc because v4–v8 established the failure is structural:

*Anti-dithering*: new `torque_saturation` penalty (−5, mean of
max(0, |τ|/limit − 0.9) — prices time-at-saturation, which `dof_torques_l2`
cannot see since magnitude caps at the limit); `action_rate_l2` −0.01 → −0.2
(×20).

*Anti-hop*: new `feet_airborne_too_long` penalty (−1 per foot per step over
1 s of current air time, gated moving) — the tripod-hop carried legs that
never landed, invisible to both the idle-gated feet term and the
touchdown-priced air-time floor.

*Observability*: binary foot contacts added to the policy obs (69 → 75) —
the gait rewards key on contact events the policy previously could not see.

*Audit removals*: `joint_deviation_l1` (anchors to the standing pose, fights
stride); `flat_orientation_l2` (triple-covered orientation with
`hip_height_variance` + `ang_vel_xy_l2`); `track_yaw_rate` merged into
`track_heading` at 2.5 (the yaw-rate command is a deterministic P-function of
heading error — the pair rewarded one channel at two derivatives).

Expected signature: saturation/action-rate costs shrink first (dithering
unlearned, tracking temporarily worse), THEN gait terms move. If v9 still
produces no gait, v10 adds a gait phase clock (explicit prior — kept out of
v9 so attribution stays clean).

**v9 outcome — every targeted pathology fixed; walking still did not emerge**
(run `2026-09-13_01-25-06`, stopped at ~iteration 2100 of 2500 on rollout
review): the smoothness package worked exactly as designed — worst joint
saturation fell 98% → ~50-60% and kept falling, action-rate cost shrank
monotonically (−0.56 → −0.16), `feet_airborne_too_long` confirmed carried
legs fading, `stance_progress` climbed monotonically to +0.012 (~5× v8),
and `feet_air_time` finally **crossed zero (+0.0004 at iter 2000)** in the
predicted smoothness-first order. But visually the policy spends its skill
on *reorienting* (turn-in-place is crisp) while translation remains
micro-stepped creep — technically-positive air time, no real strides. Five
walking versions deep, the conclusion is that the full task (heading + speed
+ height + idle gating + 20 terms) has too many competing objectives to
learn locomotion inside; reorienting alone satisfies most of the reward
mass. **Next: isolate locomotion in a stripped-down task** (see March below)
before returning to the full command set; gait clock still reserved as the
following step if needed.

## Iteration 4 — `Spidertron-March-v0`: locomotion isolated (2026-09-13)

Diagnostic task after the v9 tap-dance finding: fixed heading/speed
(0.2 m/s), fixed height, no idle mode, ±0.3 spawn yaw, plus the
**load-coupling** term `foot_duty_deviation` (every foot's contact duty in
[0.45, 0.75] while moving) and live gait-structure probes
(`Curriculum/metric_foot_duty_min|mean` — duty of the least-loaded foot is
the tap-dance detector).

**Run 1** (1000 iters / 2 h 44 min, run `2026-09-13_03-25-39`): the
load-coupling concept works — the policy first dove to duty_min 0.04 (the
old unequal-load instinct), then monotonically bought the penalty back:
**duty_min 0.04 → 0.22, duty_mean centered at 0.51, stance_progress 0.050
(4× the v9 walk's best), forward tracking 2.82/3.0, zero falls.** Plateaued
from ~iter 650: some legs stuck near 0.2 duty, and swings remain under the
0.15 s air-time floor (fast shallow cycles — feet_air_time −0.026).
**Run 2** (resumed from model_999): `foot_duty` −2 → −4 and `feet_air_time`
6 → 8 — duty_min stepped up to a 0.27–0.29 plateau, but swings kept
*shrinking* as duty rose: the two terms were structural antagonists, because
the 0.15 s floor / 0.4 s target band sat entirely above the ~0.13–0.14 s
swing the policy converged to in every configuration since walk v4 — the
plant's natural cadence (cf. √(l/g) ≈ 0.17 s).

**Run 3 — align the band with the revealed cadence** (floor 0.08 / target
0.25, resumed from r2 model_1350): `feet_air_time` went positive
immediately (earnable for the first time in nine runs) and **both gait
channels climbed together** — duty_min 0.30 → **0.346 peak at iteration
~1600** before drifting; stopped there. **Peak-checkpoint result
(model_1600, archived with video + curves in docs/): duty_min 0.35 (17×
the v9 walk's tap-dance value), duty_mean 0.59, air-time ≈ break-even,
forward tracking 2.81/3.0 at 0.2 m/s, zero falls.** Declared a good-enough
march; promoted to the walk port.

**v4 — tripod shaping** (user review: robots mainly hop on flat ground;
rough terrain would select against hops, flat does not): `tripod_antiphase`
(+1.5, |mean_contact(A) − mean_contact(B)| — the alternating-tripod sets,
derived from mount geometry as the unique adjacent-free 2-coloring of the
leg ring, so no body orientation is prescribed) and `too_many_feet_airborne`
(−2 per foot beyond 3). Speed 0.2 → 0.25, swing target 0.3, clearance 0.05.
Mid-v4 a **20× throughput collapse** was diagnosed and fixed
(`ContactSensor.body_names` materializing 78k prim paths per in-loop call —
also the silent 4× tax on every run since march r1; commit 122c294).
**v4b result** (resume of r3 model_1600 → iter 2500): tripod metric
0.17 → **0.39**, airborne cost −2.4 → −0.6 (hop ~75% abandoned), duty_min
0.36, tracking 2.76/3.0 at 0.25 m/s.

**v5 — exact-3 + within-tripod balance** (user-directed):
`contact_count_deviation` (−2, two-sided |contacts−3|) and
`tripod_contact_time_balance` (−5, within-group contact-time variance).
Resumed from v4b model_2500 → iter 3500. contact_count improved steadily
(−1.02 → −0.60) but the tripod metric **plateaued at 0.32–0.33**
(+0.01/300 iters, below the extension bar) with balance flat (−0.31) —
the six-resume chain appears habit-locked: each new constraint now trades
against the previous ones instead of compounding. Final: reward 2.90,
tracking 2.71, duty_min 0.28. **Next: fresh from-scratch run under the
complete v5 reward set** (all constraints present from the first gradient
step, no hop history to unlearn) before considering the gait clock.

**v6 — fresh from-scratch under the complete v5 reward set** (run
`2026-09-13_14-51-18`, 2000 iters): the habit-lock hypothesis confirmed
decisively — tripod antiphase hit **0.92 within 300 iterations** (the
six-resume chain peaked at 0.39) and held ~0.93 throughout. Final:
**duty_min 0.456** (project record), tripod_balance −0.06 (within-group
equality essentially satisfied), contact_count −0.37, air-time positive,
tracking 2.79/3.0, zero falls across the entire run. Step lengths 7–12 cm.
Remaining asymmetry: a stable carry/stride role split between the tripods
(~85% vs ~10% duty) — internally balanced, cleanly anti-phased, but not
symmetric alternation; duty_min's window-average (0.46) vs the rollout
table's per-foot duty (~10%) suggests the light tripod runs frequent short
cycles. Artifacts: docs/march_v6_spidertron.mp4, march_v6_training.png,
march_v6_gait_diagnostics.png, policies/spidertron_march_v6_model_1999.pt.

**The load-bearing lesson of the arc**: constraints compound when present
from the first gradient step and merely trade when retrofitted onto a
converged policy — fresh runs under the full reward set beat every
resume-chain attempt by 2–3× on the coordination metrics at a fraction of
the compute.

**March-max variant** (`Spidertron-MarchMax-v0`, in progress): identical
constraint suite, linear (uncapped) forward-velocity reward — measures the
fastest honest tripod the constraints permit. First attempt converged to a
0.85 s lunge-and-fall (early termination SAVED accumulated penalties; the
fall was free) — fixed with `is_terminated` −200 and speed weight 8 → 5.
Relaunch: zero falls, ~0.4 m/s average within 200 iterations with
antiphase ~0.65–0.68.

## The speed-constraint spectrum (arc capstone, 2026-09-13)

Three runs, same speed incentive, three constraint regimes:

| | v6: full prior, setpoint | max2: full prior, max speed | free: no prior |
|---|---|---|---|
| reward for speed | exp tracking @ 0.25 m/s | linear, uncapped | linear, uncapped |
| gait terms | all | all | **none** |
| achieved speed | 0.25 m/s | **2.13 m/s** | ~41 m/s (unphysical) |
| tripod antiphase | 0.93 | 0.55 | 0.13 |
| duty (min / typical) | 0.46 / ~0.5 | 0.21 / ~0.3 | — |
| falls | 0 | 0 | ~2% (suppressed by −400) |
| energy audit | honest | honest | **3.3× solver injection** |
| character | clean tripod walk | tripod-flavored run, 25–50 cm strides | contact-solver paddle-wheel |

**max2 final** (run `2026-09-13_16-12-15`, 2000 iters / 1 h 37 min): 2.13 m/s
mean — 8.5× the walking setpoint — with all six feet cycling (17–42% duty,
3–7.5 steps/s, 24–51 cm strides), antiphase 0.55, zero falls. The fastest
honest gait the constraint suite permits at these weights; its cost
breakdown (contact_count −2.69, foot_duty −1.91 continuously paid) prices
the static-stability prior at speed. Artifacts:
docs/march_max_spidertron.mp4, march_max_gait_diagnostics.png,
policies/spidertron_march_max_model_1999.pt. Note the first max attempt
found a different exploit — lunge-and-fall, since early termination SAVED
accumulated penalties — fixed by is_terminated −200.

**free-run forensics** (stopped at ~500 iters; scripts/analyze_free_exploit.py):
speed ratcheted +1.7 m/s per ~4 Hz ground-brush, linearly and unboundedly
(41 m/s at eval, heading for the 100 m/s URDF cap). Not slamming (peak force
only 4× bodyweight), not flight (body skims at 0.15–0.2 m): joints are
BACKDRIVEN to 22.8 rad/s — 4× the drive's velocity limit, which caps the
motor but not external contact — and the stiff implicit position-drive
fighting the backdriven joint at the solver level transfers unphysical
momentum each contact. Energy audit: KE 2916 J vs 872 J integrated actuator
work. The dithering exploit's final form: v8 saturation-averaged position
control; free solver-pumped propulsion.
Artifacts: docs/march_free_exploit_spidertron.mp4, march_free_exploit_timeline.png.

**Capstone lessons**: (1) the gait constraint suite is what keeps the
optimizer inside physically-meaningful dynamics on flat ground — foot_slip/
duty/air-time fence off exactly the solver's unphysical region (real-world
physics polices this for free; sim reward must). (2) Never let early
termination be profitable: price death above anything an episode can earn.
(3) The honest speed range of this platform under the tripod prior:
0.25–2.1 m/s, with structure degrading smoothly (antiphase 0.93 → 0.55) as
speed rises.

**March → walk transfer (walk v10, in config)**: `foot_duty_deviation`
(−4) added to the walk rewards; swing band aligned (0.08/0.25);
`stance_progress` 0.5 → 1.0; the falsified v8 speed ramp replaced by the
duty min/mean probes. Everything else (v9 smoothness package, contact obs,
strict-facing command) unchanged.

**v8 (as configured)**: two structural changes. (1) **Speed curriculum**
(`mdp/curriculums.py command_speed_ramp`): command range starts 0–0.1 m/s and
widens linearly to 0–0.3 by iteration ~1250 — at low speed, stepping barely
disturbs tracking, so the barrier is thin; learn the gait there, carry it up.
Disabled in the PLAY cfg (fresh env would be capped at 0.1). (2)
**`stance_progress` reward** (+0.5): forward speed × count of feet in contact
with world-frame speed < 5 cm/s — pays the stride itself; skating feet fail
the slip tolerance, standing pays zero forward speed.
