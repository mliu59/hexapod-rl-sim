# Training observability

One wrapper around the training loop, plus probes that watch it. Adding a tool
means writing a probe; training scripts do not change.

```python
from hexapod_lab.observability import training_session

with training_session(log_dir=log_dir, task=..., num_envs=..., max_iterations=...):
    runner.learn(...)
```

## Why a scaffold rather than edits in `train.py`

rsl-rl's `OnPolicyRunner.learn()` has no per-iteration callback, so instrumentation
has to come from either a sampler thread or the event files after the run. Both
patterns fit one `start()`/`stop()` lifecycle, so they live behind one interface
instead of being scattered through the training script. Monkeypatching `learn()`
would be the alternative and it breaks on every rsl-rl upgrade.

**Probes never take down a run.** Every probe call is wrapped; a probe that raises
prints a traceback and training continues. Hours of compute outrank any
measurement. If the *training loop* raises, the session records the exception on
`ctx.error` and probes still get `stop(failed=True)`, so a crashed run leaves
`error.txt`, a `console.log` ending in the traceback, and whatever partial
reports could be built — which is when you most want them.

## Ordering

Probes start in ascending `start_order` and stop in ascending `stop_order`. Two
independent numbers, because the constraints genuinely do not nest: the console
probe must start *first* and stop *last*, while the report probe must stop
*after* the VRAM probe has closed its event file (read a half-written file and
the report silently loses the tail of the run).

| Kind | `start_order` | `stop_order` |
| --- | --- | --- |
| context capture (console) | 0 | 90 |
| live samplers (VRAM) | 10 | 10 |
| post-hoc readers (reports) | 10 | 20 |

## What console capture does and does not get

`ConsoleLogProbe` tees at the Python level, so it captures rsl-rl's iteration
tables, Isaac Lab's manager dumps (the only place obs/action dimensions are
printed), warnings, probe output and tracebacks.

It does **not** capture Kit and carb's native `[Warning] [carb…]` lines — those
are written from C++ straight to the OS file descriptors. Capturing them needs
`os.dup2` redirection of fd 1/2 inside a live Kit process, which risks breaking
the runs this exists to observe. Redirect the whole command when you need them:

```bash
python -u scripts/rsl_rl/train.py ... > run.log 2>&1
```

Also uncaptured: anything printed before `training_session` starts (Kit boot,
URDF→USD conversion), and `logging` handlers bound to the original `sys.stderr`
at import time.

## What ships today

| Probe | Kind | What it does |
| --- | --- | --- |
| `ConsoleLogProbe` | wrapping | Tees stdout/stderr to `<run_dir>/console.log`; writes `error.txt` on a crash |
| `VramProbe` | live | Samples GPU memory every 10 s into the run's TensorBoard dir (`GPU/vram_*`); prints peak and headroom at exit |
| `RewardReportProbe` | post-hoc | Renders PNG reports and `summary.md` from the event files |

Artifacts land in the run directory:

- `console.log` — everything Python printed, ANSI colour stripped
- `error.txt` — traceback, only present if the run raised

and in `<run_dir>/reports/`:

- `overview.png` — reward, episode length, throughput, VRAM
- `reward_terms.png` — **one panel per reward term**, green reward / red penalty
- `reward_balance.png` — rewards stacked up, penalties stacked down, net line
- `ppo_diagnostics.png` — losses, learning rate, action-noise std, tracking error
- `summary.md` — metadata, headline scalars, reward terms ranked by magnitude
- `run_summary.json` — machine, config and probe results

## Regenerating reports

Reports are pure functions of the event files, so they can be rebuilt any time —
including for runs recorded before this module existed. No Isaac Sim required.

```bash
python scripts/report_run.py logs/rsl_rl/hexapod_flat/<run>
python scripts/report_run.py --all
```

## Adding a probe

1. Subclass `Probe` in `hexapod_lab/observability/probes/`.
2. Set `name`; implement `start()` for live work, `stop()` for post-hoc work.
   Set `start_order` / `stop_order` if the probe has ordering constraints.
3. Export it from `probes/__init__.py` and add it to `default_probes()` in
   `session.py`.
4. Put anything worth keeping into `ctx.results` — the session writes that to
   `run_summary.json`. Check `ctx.error` if the probe should behave differently
   on a crashed run.

```python
class MyProbe(Probe):
    name = "my_probe"

    def start(self, ctx): ...
    def stop(self, ctx, failed=False):
        ctx.results["something_measured"] = 42
```

Write artifacts under `ctx.reports_dir` so a run directory stays self-describing.

## Deliberately not logged

**GPU utilization percentage.** `nvidia-smi`'s number reports "at least one kernel
was resident", not how much of the card was busy — it reads high on
launch-bound workloads while most of the hardware idles. `Perf/total_fps`
(already logged by rsl-rl) answers the question that actually matters.

## Ideas not yet built

- Gait diagnostics: contact/air-time raster per leg, duty factor, tripod
  detection — the reward curve cannot tell a tripod from a shuffle
- Command-vs-achieved velocity scatter, split by command bin
- Automatic rollout video capture at checkpoint intervals
- Cross-run comparison: overlay reward curves from several runs on one axis
- Action saturation histogram (how often the policy pins joints at limits)
