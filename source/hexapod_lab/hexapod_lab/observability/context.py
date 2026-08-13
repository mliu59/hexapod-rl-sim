"""Run metadata shared by every probe."""

from __future__ import annotations

import contextlib
import platform
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    """Everything a probe might need to know about the run it is watching.

    ``log_dir`` is the single source of truth: probes write their artifacts under
    it so a run directory stays self-describing. Anything a probe learns that is
    worth carrying forward goes into ``results``, which the session writes out as
    ``run_summary.json`` at the end.
    """

    log_dir: str
    task: str = "unknown"
    num_envs: int = 0
    max_iterations: int = 0
    device: str = "unknown"
    run_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    #: set by the session when the training loop raises: {"type", "message", "traceback"}
    error: dict[str, str] | None = None

    @property
    def reports_dir(self) -> Path:
        path = Path(self.log_dir) / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def describe_machine(self) -> dict[str, Any]:
        """Host details worth pinning to a run -- config-to-result mapping does not
        survive memory, and neither does 'which machine was that on'."""
        info: dict[str, Any] = {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
        try:
            import torch

            info["torch"] = torch.__version__
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["gpu_total_gb"] = round(torch.cuda.mem_get_info()[1] / 1024**3, 2)
        except Exception:  # noqa: BLE001 - metadata must never break a run
            pass
        with contextlib.suppress(Exception):  # metadata must never break a run
            info["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                timeout=10,
            ).strip()
        return info
