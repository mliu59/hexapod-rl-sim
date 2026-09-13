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
```

Registered tasks: `Spidertron-Stand-v0`, `Spidertron-HeightTrack-v0`,
`Spidertron-Walk-v0` (each with a `-Play-v0` variant), plus the earlier
placeholder-robot `Hexapod-Flat-v0`.

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
- [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) — the training observability
  scaffold
- [cad/README.md](cad/README.md) — CAD pipeline

## Progress

Stand and dynamic height-tracking are trained, visually verified, and
committed (millimeter steady-state, sub-0.3 s settling on setpoint steps;
final policy incl. ONNX in [docs/policies/](docs/policies/)). Commanded
walking (heading + speed + height) is mid-development: heading and turn-rate
tracking converge; gait quality is the open question, iterating on reward
shaping. Rough terrain is next.
