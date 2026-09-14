"""Rigid-body mass properties from solid geometry.

The one job here: turn geometry into the numbers a URDF ``<inertial>`` block
needs, without anyone guessing. Everything flows through OpenCascade's
``GProp``/``BRepGProp`` volume integration, which is exact for B-rep solids
(not a mesh approximation).

Units
-----
build123d models in this project are authored in **millimetres** — that is the
library's own convention and fighting it invites silent 1000x errors. URDF
requires **metres and kilograms**. The conversion happens exactly once, here,
on the way out. Every :class:`MassProperties` instance is SI; nothing else in
the codebase should be doing unit maths on mass properties.

Inertia convention
------------------
:attr:`MassProperties.inertia` is the 3x3 tensor **about the centre of mass**,
in the model's axis orientation. That is the convention URDF wants when the
``<origin>`` of the ``<inertial>`` element is placed at the COM, which is what
we do. Use :meth:`MassProperties.about` to re-reference it to another point via
the parallel-axis theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from build123d import Shape, import_step
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from hexapod_cad.materials import Material
from hexapod_cad.materials import get as get_material

# --- unit conversions, applied once on the way out of OCCT -----------------
_MM3_TO_M3 = 1.0e-9
_MM_TO_M = 1.0e-3
# OCCT integrates with unit density, so its "matrix of inertia" carries units of
# volume x length^2 = mm^5. Scaling to m^5 then multiplying by kg/m^3 yields kg*m^2.
_MM5_TO_M5 = 1.0e-15


@dataclass(frozen=True)
class MassProperties:
    """Rigid-body properties of a part or a composed link, in SI units."""

    mass: float
    """kg."""

    com: np.ndarray
    """Centre of mass, metres, shape (3,), in the frame the geometry was built in."""

    inertia: np.ndarray
    """Inertia tensor about the centre of mass, kg*m^2, shape (3, 3)."""

    volume: float
    """m^3. Kept because it is the audit trail for ``mass = volume * density``."""

    label: str = ""

    def __post_init__(self) -> None:
        if self.mass < 0.0:
            raise ValueError(f"{self.label!r}: negative mass {self.mass}")
        if self.com.shape != (3,):
            raise ValueError(f"{self.label!r}: com must be shape (3,), got {self.com.shape}")
        if self.inertia.shape != (3, 3):
            raise ValueError(f"{self.label!r}: inertia must be shape (3, 3), got {self.inertia.shape}")

    def about(self, point: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        """Inertia tensor about ``point`` (metres, same frame as :attr:`com`).

        Parallel-axis theorem: ``I_p = I_com + m * (|d|^2 * E - d d^T)``.
        """
        d = np.asarray(point, dtype=float) - self.com
        return self.inertia + self.mass * (float(d @ d) * np.eye(3) - np.outer(d, d))

    def translated(self, offset: np.ndarray | tuple[float, float, float]) -> MassProperties:
        """Same body, rigidly moved by ``offset`` metres. Inertia is unchanged."""
        return MassProperties(
            mass=self.mass,
            com=self.com + np.asarray(offset, dtype=float),
            inertia=self.inertia,
            volume=self.volume,
            label=self.label,
        )

    def is_physically_valid(self, tol: float = 1e-12) -> bool:
        """Check the tensor is symmetric, positive-definite, and obeys the triangle inequality.

        A tensor failing this cannot describe a real rigid body, and PhysX will
        either reject it or quietly produce nonsense. Worth asserting before
        anything reaches a URDF.
        """
        if not np.allclose(self.inertia, self.inertia.T, atol=1e-9):
            return False
        eig = np.linalg.eigvalsh(self.inertia)
        if np.any(eig <= tol):
            return False
        a, b, c = sorted(eig)
        return a + b >= c - tol

    def __repr__(self) -> str:
        ixx, iyy, izz = np.diag(self.inertia)
        return (
            f"MassProperties({self.label!r}, mass={self.mass * 1e3:.2f} g, "
            f"com=({self.com[0] * 1e3:.2f}, {self.com[1] * 1e3:.2f}, {self.com[2] * 1e3:.2f}) mm, "
            f"diag(I)=({ixx:.3e}, {iyy:.3e}, {izz:.3e}) kg*m^2)"
        )


def _integrate(shape: Shape) -> tuple[float, np.ndarray, np.ndarray]:
    """Raw OCCT volume integration, in model units (mm).

    Returns ``(volume_mm3, com_mm, inertia_mm5_about_com)``.
    """
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, props)

    volume = props.Mass()  # unit density -> "mass" is volume
    c = props.CentreOfMass()
    com = np.array([c.X(), c.Y(), c.Z()], dtype=float)

    m = props.MatrixOfInertia()
    inertia = np.array([[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)], dtype=float)

    # OCCT's MatrixOfInertia() is already referenced to the centre of mass, not
    # to the origin, so no parallel-axis shift belongs here. Determined
    # empirically, not from the docs: checks/check_massprops.py case [2] moves a
    # box off the origin, which is the only configuration that can tell the two
    # conventions apart (for a centred box they are identical). If OCCT ever
    # changes this, case [2] fails loudly rather than silently inflating every
    # link inertia in the URDF.
    return volume, com, inertia


def of_solid(shape: Shape, material: Material | str, label: str = "") -> MassProperties:
    """Mass properties of an authored build123d solid, at ``material``'s density.

    ``shape`` is assumed to be in millimetres (build123d's convention).
    """
    mat = get_material(material) if isinstance(material, str) else material
    volume_mm3, com_mm, inertia_mm5 = _integrate(shape)

    volume = volume_mm3 * _MM3_TO_M3
    return MassProperties(
        mass=mat.mass_of(volume),
        com=com_mm * _MM_TO_M,
        inertia=inertia_mm5 * _MM5_TO_M5 * mat.density,
        volume=volume,
        label=label or f"<{mat.name} solid>",
    )


def of_step(
    path: str | Path,
    material: Material | str,
    datasheet_mass: float | None = None,
    label: str = "",
) -> MassProperties:
    """Mass properties of a vendor STEP part.

    Vendor CAD is frequently a simplified outer envelope — a servo modelled as a
    solid block, or conversely a shell with no internals. ``volume * density``
    is then wrong even though the geometry is real. Pass ``datasheet_mass`` (kg)
    to trust the manufacturer's figure instead: the inertia tensor is rescaled
    to that mass, which keeps the *distribution* from the geometry while fixing
    the *magnitude*. That is the best available answer short of tearing a servo
    down, and it is what the datasheet is for.
    """
    shape = import_step(str(path))
    props = of_solid(shape, material, label=label or Path(path).stem)
    if datasheet_mass is None:
        return props

    if props.mass <= 0.0:
        raise ValueError(f"{props.label!r}: STEP integrated to zero mass; is it a solid, not a shell?")
    scale = datasheet_mass / props.mass
    return MassProperties(
        mass=datasheet_mass,
        com=props.com,
        inertia=props.inertia * scale,
        volume=props.volume,
        label=props.label,
    )


def compose(parts: list[MassProperties], label: str = "") -> MassProperties:
    """Compose several rigid bodies into one, about their combined centre of mass.

    Each part's :attr:`~MassProperties.com` must already be expressed in the
    common (link) frame — use :meth:`MassProperties.translated` to place them.
    This is how a link made of a printed bracket *plus* the servo it carries
    gets a single correct inertial block.
    """
    if not parts:
        raise ValueError("compose() needs at least one part")

    total_mass = sum(p.mass for p in parts)
    if total_mass <= 0.0:
        raise ValueError("composed mass is zero")

    com = sum((p.mass * p.com for p in parts), start=np.zeros(3)) / total_mass
    inertia = sum((p.about(com) for p in parts), start=np.zeros((3, 3)))

    return MassProperties(
        mass=total_mass,
        com=com,
        inertia=inertia,
        volume=sum(p.volume for p in parts),
        label=label or f"<compound of {len(parts)}>",
    )
