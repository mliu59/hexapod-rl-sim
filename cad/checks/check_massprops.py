"""Verify mass-property extraction against closed-form rigid-body inertias.

This is the check that makes the inertia policy in ``cad/README.md``
trustworthy: if OCCT's tensor convention, the mm->m conversion, or the
parallel-axis shift is wrong, every link in the URDF is wrong by a constant
factor and nothing downstream will notice. Analytic solids are the only
independent source of truth available.

The off-origin box is the load-bearing case. A box centred at the origin has
the same tensor whether OCCT reports about the origin or about the centre of
mass, so it cannot distinguish the two conventions; an offset box can.

Run:
    cd cad && uv run python checks/check_massprops.py

(Lives in ``checks/`` rather than ``tests/`` because the repo's root
``.gitignore`` ignores ``tests/`` at any depth — an artifact of the Isaac Lab
template this repo was scaffolded from.)
"""

from __future__ import annotations

import sys

import numpy as np
from build123d import Box, Cylinder, Pos, Sphere
from hexapod_cad import compose, of_solid
from hexapod_cad.materials import get as get_material

RTOL = 1e-6

_failures: list[str] = []
_checks = 0


def check(name: str, got: float | np.ndarray, want: float | np.ndarray, rtol: float = RTOL) -> None:
    global _checks
    _checks += 1
    got_a, want_a = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if np.allclose(got_a, want_a, rtol=rtol, atol=1e-15):
        print(f"  ok   {name}")
        return
    err = np.max(np.abs(got_a - want_a) / np.maximum(np.abs(want_a), 1e-30))
    print(f"  FAIL {name}: got {got_a!r} want {want_a!r} (max rel err {err:.3e})")
    _failures.append(name)


def diag(ixx: float, iyy: float, izz: float) -> np.ndarray:
    return np.diag([ixx, iyy, izz])


def main() -> int:
    al = get_material("al6061")
    rho = al.density

    # --- 1. box centred at the origin ---------------------------------------
    # Dimensions in mm; analytic inertia needs metres.
    lx, ly, lz = 100.0, 60.0, 20.0
    mx, my, mz = lx / 1e3, ly / 1e3, lz / 1e3
    print("\n[1] box 100x60x20 mm, al6061, centred at origin")
    p = of_solid(Box(lx, ly, lz), al, label="box")
    vol = mx * my * mz
    mass = vol * rho
    check("volume", p.volume, vol)
    check("mass", p.mass, mass)
    check("com", p.com, np.zeros(3))
    check(
        "inertia",
        p.inertia,
        diag(
            mass * (my**2 + mz**2) / 12.0,
            mass * (mx**2 + mz**2) / 12.0,
            mass * (mx**2 + my**2) / 12.0,
        ),
    )
    check("physically valid", float(p.is_physically_valid()), 1.0)
    box_inertia_about_com = p.inertia.copy()

    # --- 2. the same box, moved off the origin ------------------------------
    # Discriminates "tensor about origin" from "tensor about COM". If the shift
    # in _integrate() were missing, this inertia would be inflated by the
    # parallel-axis term and only this case would catch it.
    offset_mm = np.array([250.0, -120.0, 40.0])
    print(f"\n[2] same box translated to {tuple(offset_mm)} mm (convention discriminator)")
    p2 = of_solid(Pos(*offset_mm) * Box(lx, ly, lz), al, label="box-offset")
    check("mass unchanged", p2.mass, mass)
    check("com", p2.com, offset_mm / 1e3)
    check("inertia about com unchanged", p2.inertia, box_inertia_about_com)
    # ...and the parallel-axis helper must reproduce the origin tensor.
    d = offset_mm / 1e3
    check(
        "about(origin) == parallel-axis",
        p2.about((0.0, 0.0, 0.0)),
        box_inertia_about_com + mass * (float(d @ d) * np.eye(3) - np.outer(d, d)),
    )

    # --- 3. sphere -----------------------------------------------------------
    r_mm = 25.0
    r = r_mm / 1e3
    print(f"\n[3] sphere r={r_mm} mm")
    ps = of_solid(Sphere(r_mm), al, label="sphere")
    vol_s = 4.0 / 3.0 * np.pi * r**3
    mass_s = vol_s * rho
    check("volume", ps.volume, vol_s, rtol=1e-4)  # tessellation-free but OCCT is still numeric
    check("mass", ps.mass, mass_s, rtol=1e-4)
    i_s = 2.0 / 5.0 * mass_s * r**2
    check("inertia", ps.inertia, diag(i_s, i_s, i_s), rtol=1e-4)

    # --- 4. cylinder (axis +z) ----------------------------------------------
    rc_mm, h_mm = 15.0, 80.0
    rc, h = rc_mm / 1e3, h_mm / 1e3
    print(f"\n[4] cylinder r={rc_mm} mm h={h_mm} mm, axis +z")
    pc = of_solid(Cylinder(rc_mm, h_mm), al, label="cyl")
    vol_c = np.pi * rc**2 * h
    mass_c = vol_c * rho
    check("volume", pc.volume, vol_c, rtol=1e-4)
    check(
        "inertia",
        pc.inertia,
        diag(
            mass_c * (3.0 * rc**2 + h**2) / 12.0,
            mass_c * (3.0 * rc**2 + h**2) / 12.0,
            mass_c * rc**2 / 2.0,
        ),
        rtol=1e-4,
    )

    # --- 5. compose() two halves back into the whole ------------------------
    # End-to-end check of the composition path a real link uses: two separately
    # measured bodies, placed in a common frame, must reproduce the monolith.
    print("\n[5] compose(): two 50x60x20 halves == one 100x60x20 box")
    half = Box(lx / 2, ly, lz)
    left = of_solid(Pos(-lx / 4, 0, 0) * half, al, label="left")
    right = of_solid(Pos(lx / 4, 0, 0) * half, al, label="right")
    both = compose([left, right], label="halves")
    check("mass", both.mass, mass)
    check("volume", both.volume, vol)
    check("com", both.com, np.zeros(3))
    check("inertia", both.inertia, box_inertia_about_com)

    # --- 6. translated() is inertia-preserving ------------------------------
    print("\n[6] translated() moves com, preserves inertia about com")
    moved = p.translated((0.1, 0.2, 0.3))
    check("com", moved.com, np.array([0.1, 0.2, 0.3]))
    check("inertia", moved.inertia, box_inertia_about_com)

    # --- summary -------------------------------------------------------------
    print(f"\n{'-' * 60}")
    if _failures:
        print(f"FAILED {len(_failures)}/{_checks}: {', '.join(_failures)}")
        return 1
    print(f"all {_checks} checks passed")
    print("\nsample repr:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
