"""Tee stdout/stderr into the run directory, and record crashes.

A run directory is supposed to be self-describing, but until this probe existed
the most diagnostic output of all -- the rsl-rl iteration tables, Isaac Lab's
manager dumps (the only place the observation and action dimensions are
printed), warnings, and any crash traceback -- went to the terminal and vanished
unless someone remembered to redirect. Hydra's ``hydra.log`` captures none of it.

Writes:

* ``<run_dir>/console.log`` -- everything Python printed, ANSI colour stripped
* ``<run_dir>/error.txt``   -- traceback, written only if the run raised

**Known limitation, by design.** This tees at the Python level
(``sys.stdout`` / ``sys.stderr``). Kit and carb emit their log lines from C++
straight to the OS file descriptors, so *native simulator output is not
captured* -- the ``[Warning] [carb...]`` lines stay terminal-only. Catching
those needs ``os.dup2`` redirection of fd 1/2 inside a live Kit process, which
risks breaking the very runs this is meant to observe. Probes must never take
down a run, so the safe 90% is the right trade. Redirect the whole command if
you need the native lines too::

    python -u scripts/rsl_rl/train.py ... > run.log 2>&1

Two smaller caveats: anything printed before ``training_session`` starts (Kit
boot, asset conversion) is not captured, and ``logging`` handlers bound to the
original ``sys.stderr`` at import time keep writing there.
"""

from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path
from typing import TextIO

from ..context import RunContext
from ..probe import Probe

# rsl-rl prints bold/colour codes around its iteration tables; strip them so the
# file stays greppable
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class _Tee:
    """Write to the real stream and to a file. Deliberately minimal."""

    def __init__(self, stream: TextIO, sink: TextIO) -> None:
        self._stream = stream
        self._sink = sink

    def write(self, data: str) -> int:
        written = self._stream.write(data)
        with contextlib.suppress(Exception):  # never let logging break the run
            self._sink.write(ANSI_RE.sub("", data))
        return written

    def flush(self) -> None:
        self._stream.flush()
        with contextlib.suppress(Exception):
            self._sink.flush()

    # pass-throughs for libraries that interrogate the stream
    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    def writable(self) -> bool:
        return True

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


class ConsoleLogProbe(Probe):
    """Capture console output for the duration of the run, plus any traceback."""

    name = "console"
    start_order = 0  # first, so everything after it is captured
    stop_order = 90  # last, so the other probes' shutdown output is captured too

    def __init__(self, filename: str = "console.log") -> None:
        self.filename = filename
        self._sink = None
        self._stdout = None
        self._stderr = None

    def start(self, ctx: RunContext) -> None:
        path = Path(ctx.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        # line buffered so a hard kill still leaves a usable log. Deliberately not
        # a context manager: the handle has to stay open from start() to stop().
        self._sink = open(  # noqa: SIM115
            path / self.filename, "a", encoding="utf-8", errors="replace", buffering=1
        )
        self._sink.write(f"=== run start: {ctx.task} | {ctx.num_envs} envs | {ctx.max_iterations} iters ===\n")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._stdout, self._sink)
        sys.stderr = _Tee(self._stderr, self._sink)
        print(f"[console] teeing stdout/stderr to {path / self.filename}")

    def stop(self, ctx: RunContext, failed: bool = False) -> None:
        if self._sink is None:
            return
        if ctx.error:
            # the traceback reaches stderr anyway, but only if the exception
            # propagates to the top; recording it here means a crashed run is
            # diagnosable from its own directory
            error_path = Path(ctx.log_dir) / "error.txt"
            error_path.write_text(
                f"{ctx.error['type']}: {ctx.error['message']}\n\n{ctx.error['traceback']}",
                encoding="utf-8",
            )
            self._sink.write(f"\n=== run FAILED ===\n{ctx.error['traceback']}\n")
            print(f"[console] run failed; traceback written to {error_path}")
            ctx.results["error_file"] = error_path.name

        self._sink.write("=== run end ===\n")
        # restore before closing, or later writes hit a closed file
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        self._sink.close()
        self._sink = None
        ctx.results["console_log"] = self.filename
