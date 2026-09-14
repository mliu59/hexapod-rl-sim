"""CAD-driven generation of the hexapod robot model.

Authors solid geometry with build123d and derives URDF link properties from it.
Nothing in this package may import Isaac Sim, Isaac Lab, or torch — it runs in
its own venv (see ``cad/README.md``).
"""

from __future__ import annotations

from hexapod_cad.massprops import MassProperties, compose, of_solid, of_step
from hexapod_cad.materials import Material, MATERIALS

__all__ = [
    "MATERIALS",
    "Material",
    "MassProperties",
    "compose",
    "of_solid",
    "of_step",
]
