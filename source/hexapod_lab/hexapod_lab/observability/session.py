"""The scaffold: one wrapper around a training loop that runs a set of probes.

Usage in a training script::

    with training_session(log_dir, task=..., num_envs=..., probes=default_probes()):
        runner.learn(...)

Adding a new observability tool means writing a Probe subclass and appending it
to the list -- training scripts do not change. That is the whole point: the
instrumentation grows without every script growing with it.

Probe failures are contained. A crashed probe prints a traceback and the run
continues; a crashed *training loop* still gets ``stop(failed=True)`` on every
probe, so partial artifacts survive a failed run (which is exactly when you most
want to look at them).
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from .context import RunContext
from .probe import Probe, safe_call
from .probes.console import ConsoleLogProbe
from .probes.reward_report import RewardReportProbe
from .probes.vram import VramProbe


def default_probes() -> list[Probe]:
    """The standard set. Extend this as tools are added."""
    return [ConsoleLogProbe(), VramProbe(), RewardReportProbe()]


@contextmanager
def training_session(
    log_dir: str,
    task: str = "unknown",
    num_envs: int = 0,
    max_iterations: int = 0,
    device: str = "unknown",
    run_name: str = "",
    probes: Sequence[Probe] | None = None,
    extra: dict | None = None,
) -> Iterator[RunContext]:
    """Wrap a training loop with observability probes."""
    ctx = RunContext(
        log_dir=log_dir,
        task=task,
        num_envs=num_envs,
        max_iterations=max_iterations,
        device=device,
        run_name=run_name,
        extra=extra or {},
    )
    active: Iterable[Probe] = list(probes) if probes is not None else default_probes()

    print(f"[observability] probes: {', '.join(p.name for p in active) or 'none'}")
    for probe in sorted(active, key=lambda p: p.start_order):
        safe_call(probe, "start", ctx)

    started = time.time()
    failed = False
    try:
        yield ctx
    except BaseException as exc:
        failed = True
        # record the crash before re-raising, so probes can persist it. A run
        # that died is the one you most want a diagnosable log from.
        ctx.error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        raise
    finally:
        ctx.results["wall_time_s"] = round(time.time() - started, 1)
        ctx.results["failed"] = failed
        for probe in sorted(active, key=lambda p: p.stop_order):
            safe_call(probe, "stop", ctx, failed)
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            (Path(log_dir) / "run_summary.json").write_text(
                json.dumps(
                    {
                        "task": ctx.task,
                        "run_name": ctx.run_name,
                        "num_envs": ctx.num_envs,
                        "max_iterations": ctx.max_iterations,
                        "device": ctx.device,
                        "extra": ctx.extra,
                        "results": ctx.results,
                        "error": ctx.error,
                        "machine": ctx.describe_machine(),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            print("[observability] could not write run_summary.json")
