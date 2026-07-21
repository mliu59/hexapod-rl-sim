"""Verify the repo .venv resolves the full Isaac stack (no Kit launch needed)."""

import importlib.util as u

import tensordict
import torch
import rsl_rl  # noqa: F401
from isaaclab.app import AppLauncher  # noqa: F401  # triggers isaacsim import

for mod in ("hexapod_lab", "isaaclab_tasks", "isaaclab_assets", "isaacsim"):
    assert u.find_spec(mod), f"cannot resolve {mod}"

print(f"venv ok: torch {torch.__version__} | cuda {torch.cuda.is_available()} | tensordict {tensordict.__version__}")
