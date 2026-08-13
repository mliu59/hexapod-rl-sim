"""VRAM sampling probe.

VRAM is a hard ceiling, not a soft one: exceed it and the run dies with an OOM
rather than slowing down. Finding that limit 40 minutes into a training run is
the expensive way to learn it. This matters more from M3 onward -- rough terrain
meshes and the height-scanner ray grid both scale with env count.

Driver-reported *and* torch-reported figures are logged. The gap between them is
simulator overhead: at 2048 envs on flat ground, torch holds ~0.08 GB against a
driver-reported 4.74 GB, i.e. >98% of GPU memory belongs to PhysX and Kit. If
memory becomes the constraint, that ratio says tuning batch size or network
width will not help.
"""

from __future__ import annotations

import threading

import torch
from torch.utils.tensorboard import SummaryWriter

from ..context import RunContext
from ..probe import Probe

DEFAULT_INTERVAL_S = 10.0


def gpu_memory_gb() -> tuple[float, float, float, float]:
    """(used, total, torch_allocated, torch_reserved) in GB.

    The first two are driver-reported -- PhysX, Kit and torch together -- not
    just torch's allocator.
    """
    free, total = torch.cuda.mem_get_info()
    return (
        (total - free) / 1024**3,
        total / 1024**3,
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
    )


class VramProbe(Probe):
    """Sample GPU memory on a wall-clock interval into the run's TensorBoard dir.

    Writes its own event file (TensorBoard merges files in a directory), so it
    stays decoupled from rsl-rl's writer. Note the x-axis is *sample index*, not
    iteration -- a sampler thread has no idea what iteration the trainer is on.
    """

    name = "vram"

    def __init__(self, interval_s: float = DEFAULT_INTERVAL_S, tag_prefix: str = "GPU") -> None:
        self.interval_s = interval_s
        self.tag_prefix = tag_prefix
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: SummaryWriter | None = None
        self._peak = 0.0

    def start(self, ctx: RunContext) -> None:
        if not torch.cuda.is_available():
            print("[vram] CUDA unavailable; probe disabled")
            return
        self._writer = SummaryWriter(log_dir=ctx.log_dir, filename_suffix=".vram")
        self._thread = threading.Thread(target=self._sample, name="vram-probe", daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        assert self._writer is not None
        step = 0
        while not self._stop.is_set():
            used, total, allocated, reserved = gpu_memory_gb()
            self._peak = max(self._peak, used)
            self._writer.add_scalar(f"{self.tag_prefix}/vram_used_gb", used, step)
            self._writer.add_scalar(f"{self.tag_prefix}/vram_free_gb", total - used, step)
            self._writer.add_scalar(f"{self.tag_prefix}/vram_torch_allocated_gb", allocated, step)
            self._writer.add_scalar(f"{self.tag_prefix}/vram_torch_reserved_gb", reserved, step)
            self._writer.flush()
            step += 1
            self._stop.wait(self.interval_s)

    def stop(self, ctx: RunContext, failed: bool = False) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0 * self.interval_s)
        _, total, _, _ = gpu_memory_gb()
        headroom = total - self._peak
        print(f"[vram] peak {self._peak:.2f} GB of {total:.2f} GB ({headroom:.2f} GB headroom)")
        ctx.results["vram_peak_gb"] = round(self._peak, 2)
        ctx.results["vram_total_gb"] = round(total, 2)
        ctx.results["vram_headroom_gb"] = round(headroom, 2)
        if self._writer is not None:
            self._writer.close()
