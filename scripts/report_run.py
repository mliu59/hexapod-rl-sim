"""Generate an observability report for any existing run directory.

The report probe runs automatically at the end of training, but reports are pure
functions of the event files, so they can be regenerated at any time -- including
for runs recorded before the observability module existed (the ANYmal-C baseline
included). Useful for re-reading an old run, or after adding a new plot.

No Isaac Sim needed: this only reads TensorBoard event files.

Usage:
    python scripts/report_run.py logs/rsl_rl/hexapod_flat/2026-08-12_18-58-55_m2_smoke_v2
    python scripts/report_run.py --all
    python scripts/report_run.py --all --root logs/rsl_rl/anymal_c_rough
"""

import argparse
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Register `hexapod_lab` as a bare package stub before importing from it. The real
# `hexapod_lab/__init__.py` does `from .tasks import *`, which pulls in isaaclab
# and ultimately `pxr` -- i.e. it only imports inside a running Kit process. The
# observability code needs none of that (event files, numpy, matplotlib), and
# reports are worth being able to build without starting a simulator.
_pkg = types.ModuleType("hexapod_lab")
_pkg.__path__ = [str(REPO / "source" / "hexapod_lab" / "hexapod_lab")]
sys.modules.setdefault("hexapod_lab", _pkg)

from hexapod_lab.observability import RunContext, build_report  # noqa: E402

DEFAULT_ROOT = REPO / "logs" / "rsl_rl"


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("events.out.tfevents*"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build observability reports from run logs.")
    parser.add_argument("run_dir", nargs="?", help="Run directory to report on.")
    parser.add_argument("--all", action="store_true", help="Report on every run under --root.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Log root for --all.")
    args = parser.parse_args()

    if args.all:
        targets = sorted(p for p in Path(args.root).rglob("*") if is_run_dir(p))
    elif args.run_dir:
        targets = [Path(args.run_dir)]
    else:
        parser.error("pass a run directory or --all")

    if not targets:
        print(f"no run directories with event files found under {args.root}")
        return

    for run in targets:
        if not is_run_dir(run):
            print(f"[skip] {run}: no event files")
            continue
        print(f"\n=== {run} ===")
        ctx = RunContext(log_dir=str(run), run_name=run.name, task=run.parent.name)
        for path in build_report(run, ctx):
            print(f"  wrote {path.relative_to(run)}")


if __name__ == "__main__":
    main()
