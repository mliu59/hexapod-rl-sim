"""Probe base class.

A probe is one observability concern with a start/stop lifecycle around the
training loop. Two flavours exist and both fit this interface:

* **live** probes do work while training runs (a sampler thread, e.g. VRAM);
* **post-hoc** probes do their work in ``stop()``, reading whatever the run left
  behind (e.g. plotting reward terms out of the TensorBoard event files).

rsl-rl's ``OnPolicyRunner.learn()`` exposes no per-iteration callback, which is
why there is no ``on_iteration`` hook here: anything per-iteration has to come
either from a sampler thread or from the event files after the fact. Adding a
hook would mean monkeypatching the runner, which breaks on every rsl-rl upgrade.

The cardinal rule: **a probe must never take down a training run.** Hours of
compute are worth more than any measurement, so the session catches everything a
probe raises and carries on.
"""

from __future__ import annotations

import traceback
from abc import ABC

from .context import RunContext


class Probe(ABC):
    """One observability concern. Subclass and override what you need."""

    #: short identifier, used in logs and in run_summary.json
    name: str = "probe"

    #: Probes start in ascending ``start_order`` and stop in ascending
    #: ``stop_order``. Two independent numbers, because the constraints do not
    #: nest: the console probe must start *first* (to capture everything after
    #: it) and stop *last* (to capture the other probes' shutdown output), while
    #: the report probe must stop *after* the VRAM probe has closed its event
    #: file -- read a half-written event file and the report silently loses the
    #: tail of the run.
    #:
    #: Conventions:
    #:   0 / 90  context capture (console)
    #:  10 / 10  live samplers (VRAM)
    #:  10 / 20  post-hoc readers (reports)
    start_order: int = 10
    stop_order: int = 10

    def start(self, ctx: RunContext) -> None:
        """Called just before training begins."""

    def stop(self, ctx: RunContext, failed: bool = False) -> None:
        """Called after training ends -- including when it raised.

        ``failed`` is True if the training loop itself errored, so a probe can
        skip work that assumes a complete run (or, conversely, dump extra
        diagnostics precisely because it crashed).
        """


def safe_call(probe: Probe, method: str, *args, **kwargs) -> None:
    """Invoke a probe method, swallowing and reporting any failure."""
    try:
        getattr(probe, method)(*args, **kwargs)
    except Exception:  # noqa: BLE001 - deliberate: observability must not kill training
        print(f"[observability] probe '{probe.name}'.{method}() failed; continuing:")
        traceback.print_exc()
