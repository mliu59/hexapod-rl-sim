# hexapod-rl-sim

Sim-only RL locomotion for a custom 18-DOF hexapod on rough terrain, built on
Isaac Sim 5.1 / Isaac Lab with RSL-RL (PPO). Full project plan: [INTRO.md](INTRO.md).

## Layout

```
assets/urdf/hexapod.urdf   placeholder robot (generated — do not hand-edit)
scripts/
  generate_hexapod_urdf.py parametric URDF generator (geometry, masses, inertias)
  rsl_rl/train.py|play.py  RSL-RL entry points
  list_envs.py             list registered Template-* tasks
  zero_agent.py            step an env with zero/random actions (sanity checks)
source/hexapod_lab/        Isaac Lab extension: tasks, robot cfg, mdp terms
  hexapod_lab/robots/      HEXAPOD_CFG (XM430-class actuator limits + PD gains)
docs/TEMPLATE_SETUP.md     upstream template setup notes (IDE, extension manager)
```

## Environment (this machine)

- Primary Python 3.11 venv: `C:\git_ws\env_isaaclab` — activate with
  `C:\git_ws\env_isaaclab\Scripts\activate`
- Repo-local `.venv`: thin wrapper for IDE/tooling. It holds only dev tools
  (ruff, pre-commit) and resolves the entire Isaac stack from `env_isaaclab`
  and the editable source trees via `.venv\Lib\site-packages\env_isaaclab.pth`
  — no duplicated install. Recreate with `python scripts\check_venv.py` as the
  smoke test. Note: root `pyproject.toml` is lint/type-check config only; the
  installable package is `source\hexapod_lab`.
- Isaac Lab clone: `C:\git_ws\IsaacLab`
- Installed via pip: `isaacsim[all,extscache]==5.1.0`, `torch==2.7.0+cu128`
- **Pin `tensordict==0.8.*`** — 0.13+ is built against a newer torch ABI and
  crashes Kit at startup with an access violation in `tensordict\_C.pyd`
- Set `OMNI_KIT_ACCEPT_EULA=YES` in the environment for headless/scripted runs
- **GPU driver must be 580.88** (R580 branch). Newer R590+ drivers (595.x/596.x)
  crash Isaac Sim 5.1's RTX renderer at startup (`rtx.scenedb.plugin`), which
  breaks viewport playback and `--video` recording. Headless training still works.

Install this extension into the venv (editable):

```powershell
python -m pip install -e source\hexapod_lab
```

## Common commands

```powershell
# regenerate the placeholder robot after parameter changes
python scripts\generate_hexapod_urdf.py

# list tasks registered by this extension
python scripts\list_envs.py

# train / play (always headless for training — see INTRO.md operating notes)
python scripts\rsl_rl\train.py --task=Template-Hexapod-Lab-v0 --headless
python scripts\rsl_rl\play.py --task=Template-Hexapod-Lab-v0 --num_envs 32
```

Baseline reference (known-good robot, from the Isaac Lab clone):

```powershell
python C:\git_ws\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py `
    --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless
```
