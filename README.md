# hexapod-rl-sim

Reinforcement-learning locomotion for a custom 18-DOF hexapod ("spidertron"),
trained entirely in simulation. The robot is parametrically CAD-designed and
exported to URDF, physically grounded in hobby-class servo limits, and trained
with PPO across thousands of parallel simulated environments on a single
consumer GPU.

The design is an homage to [Factorio](https://www.factorio.com/)'s spidertron:

![Factorio's spidertron walking through trees](docs/factorio_spidertron.gif)

*(animation from the [Factorio wiki](https://wiki.factorio.com/Spidertron);
Factorio and the spidertron are © Wube Software)*

It has always been one of my favorite sprites/entities in the game, so I
wanted to use it as the base robot for this learning project — and because
it's not yet another biped or quadruped.

![Spidertron at nominal stance](docs/spidertron_render.png)

## Current status

One consolidated policy (`Spidertron-WalkFree-v0`) covers stand, walk, and
height tracking as regions of a single command space — no locomotion modes, no
hand-designed gait rewards. Below: the policy driven live from the interactive
browser demo app (`scripts/demo_walkfree_app.py`), commanded through a
world-frame velocity joystick and a height slider.

| Fixed camera | Chase camera |
| --- | --- |
| ![WalkFree policy, fixed camera](docs/media/demo_fixed_cam.gif) | ![WalkFree policy, chase camera](docs/media/demo_chase_cam.gif) |

- **Task**: track a world-frame heading + speed command (0–1.7 m/s, with an
  internal heading controller emitting yaw-rate — the robot turns to face its
  target, no strafing) and a base-height setpoint (0.18–0.35 m).
  Observations are blind proprioception + the commands + binary foot
  contacts; control is 100 Hz position targets on 18 joints.
- **Environment**: Isaac Lab manager-based env on flat ground, physics
  grounded in hobby-servo reality — Feetech STS-class DCMotor torque/speed
  envelopes, rubber-foot friction, and contact-integrity fixes so the
  optimizer can't exploit simulator seams
  ([docs/SIM_PHYSICS_EXPLOITS.md](docs/SIM_PHYSICS_EXPLOITS.md)).
- **Training**: RSL-RL PPO, 4096 parallel envs, ~35 min on one RTX 4070 Ti.
  Rewards are three tracking terms + a termination penalty + hardware
  feasibility costs (torque saturation, joint limits, action rate, positive
  mechanical power as an energy bill). Gait structure is measured, never
  rewarded: the resulting gait is an honest, friction-limited shuffle —
  velocity error ~0.15 m/s, height error ~1 cm, falls in 0.3% of episodes.

## Stack

- **Robot**: parametric CAD model ([cad/](cad/)) → generated URDF + meshes
  ([assets/](assets/)). Feetech STS-class servo torque/velocity limits.
- **Simulation & RL**: Isaac Sim 5.1 + Isaac Lab (PhysX GPU pipeline, 4096
  envs) with RSL-RL PPO — roughly 100k+ environment steps/s on an RTX 4070 Ti;
  a full training run takes 10 minutes to 2 hours depending on the task.
- **Task framework**: Isaac Lab's manager-based configs. Tasks share one
  task-agnostic base ([spidertron_base_env_cfg.py](source/hexapod_lab/hexapod_lab/tasks/manager_based/hexapod_lab/spidertron_base_env_cfg.py));
  a new task adds a rewards configclass, optional command terms, and a
  registration entry. Policies are goal-conditioned — setpoints (height,
  direction, speed) are observations, so any external signal can drive a
  trained policy.
- **Observability**: every training run is wrapped in a scaffold that captures
  per-term reward reports, VRAM, and console logs, and rollout videos with a
  command HUD are rendered from checkpoints *while training runs*
  ([docs/OBSERVABILITY.md](docs/OBSERVABILITY.md)).

## Quickstart

```powershell
# from a venv with the Isaac stack installed (see environment notes below)
python scripts\rsl_rl\train.py --task Spidertron-Stand-v0 --headless
python scripts\rsl_rl\play.py  --task Spidertron-Stand-Play-v0 --num_envs 32

# scripted-setpoint demos for trained policies
python scripts\demo_height_staircase.py --headless --checkpoint <model.pt>
python scripts\demo_walk_rollout.py     --headless --checkpoint <model.pt>

# interactive demo app (browser UI for the WalkFree policy)
python scripts\demo_walkfree_app.py --headless
```

Registered tasks: `Spidertron-Stand-v0`, `Spidertron-HeightTrack-v0`,
`Spidertron-Walk-v0`, `Spidertron-WalkFree-v0` (each with a `-Play-v0`
variant), plus the earlier placeholder-robot `Hexapod-Flat-v0`.

### Interactive demo app

`scripts/demo_walkfree_app.py` runs the trained WalkFree policy on one robot
and serves a local web UI at `http://127.0.0.1:8765`: the main pane streams
the live sim render (fixed wide view or a chase camera), the sidebar has a
world-frame velocity joystick (hold-last-command), a height slider,
stop/reset, and command-vs-actual telemetry. UI commands are written directly
into the task's own command terms each step, so the policy sees exactly the
interface it was trained on. Runs in real time (~100 Hz control) by defaulting
to CPU physics — the GPU pipeline is launch-bound at ~20 ms/step regardless of
env count, while one robot steps in ~7 ms on CPU. Details and tuning flags in
the script docstring.

## Environment notes

Requires a Python 3.11 venv with Isaac Sim 5.1 installed via pip
(`torch 2.7 cu128`) and this repo's extension installed editable
(`pip install -e source\hexapod_lab`). Two hard pins: `tensordict==0.8.*`
(newer wheels crash Kit) and an **R580-branch GPU driver** (R590+ breaks the
RTX renderer). Always train headless. The CAD toolchain lives in its own
uv-managed venv under `cad/` and shares nothing with the Isaac stack.

## Documentation

- [INTRO.md](INTRO.md) — full project plan and phase definitions
- [docs/SPIDERTRON_TASKS.md](docs/SPIDERTRON_TASKS.md) — per-task results,
  videos, trained policies, and reward-design post-mortems
- [docs/SIM_PHYSICS_EXPLOITS.md](docs/SIM_PHYSICS_EXPLOITS.md) — the contact
  physics / simulator-exploit exploration: how an unconstrained speed
  optimizer found (and we closed) the simulator's seams, and what that says
  about where gait structure comes from
- [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) — the training observability
  scaffold
- [cad/README.md](cad/README.md) — CAD pipeline

## Progress

Stand, dynamic height-tracking, and consolidated command-following
(stand/walk/height as one policy, see **Current status** above) are trained
and visually verified; final policies incl. ONNX in
[docs/policies/](docs/policies/). Open question: gait quality — the
exploit-free optimum at these commands is a shuffle, and whether a cleaner
stepping gait emerges is being probed by measurement (duty/antiphase/slip
metrics), not reward shaping. Rough terrain (M3) is next.
