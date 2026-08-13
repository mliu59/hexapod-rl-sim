# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training observability scaffold.

One wrapper (``training_session``) around a training loop, plus a list of
``Probe`` objects that watch it. New tools are added by writing a probe, not by
editing training scripts.

See ``docs/OBSERVABILITY.md`` for how to add one.
"""

from .context import RunContext  # noqa: F401
from .probe import Probe  # noqa: F401
from .probes.console import ConsoleLogProbe  # noqa: F401
from .probes.reward_report import RewardReportProbe, build_report, load_scalars  # noqa: F401
from .probes.vram import VramProbe, gpu_memory_gb  # noqa: F401
from .session import default_probes, training_session  # noqa: F401
