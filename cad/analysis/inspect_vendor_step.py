"""Read-only inspection of vendor STEP files against datasheet expectations.

For each part: bounding box vs datasheet dimensions, solid count, raw
integrated volume, and the mass properties both raw (volume x density) and
pinned to the datasheet mass via ``of_step(..., datasheet_mass=)``. The gap
between the two is the evidence for whether the vendor CAD is a true solid,
an envelope, or a shell — recorded in each part's PROVENANCE.md.

Run:  cd cad && uv run python analysis/inspect_vendor_step.py
"""

from __future__ import annotations

from pathlib import Path

from build123d import import_step
from hexapod_cad.massprops import of_step

VENDOR = Path(__file__).resolve().parent.parent / "vendor"

PARTS = [
    # (dir, file, datasheet mass kg, expected bbox mm (case body, sans horn/lugs))
    ("sts3215", "STS3215.step", 0.055, (45.2, 24.7, 35.0)),
    ("sts3250", "STS3250.step", 0.0745, (45.22, 24.72, 35.0)),
]

# The cases are aluminium/plastic composite; density only matters for the raw
# figure we are about to distrust anyway. Use ABS-ish plastic as the base.
BASE_MATERIAL = "delrin"


def main() -> None:
    for d, fname, ds_mass, exp in PARTS:
        path = VENDOR / d / fname
        print(f"\n=== {d}: {fname} ({path.stat().st_size / 1e6:.1f} MB) ===")
        shape = import_step(str(path))
        bb = shape.bounding_box()
        size = (bb.size.X, bb.size.Y, bb.size.Z)
        solids = shape.solids()
        print(f"solids          : {len(solids)}")
        print(f"bbox            : {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} mm")
        print(f"datasheet body  : {exp[0]} x {exp[1]} x {exp[2]} mm (case only; bbox adds horn/lugs)")

        raw = of_step(path, BASE_MATERIAL, label=f"{d}-raw")
        pinned = of_step(path, BASE_MATERIAL, datasheet_mass=ds_mass, label=f"{d}-pinned")
        print(f"integrated vol  : {raw.volume * 1e6:.1f} cm3")
        print(
            f"raw mass @delrin: {raw.mass * 1e3:.1f} g   vs datasheet {ds_mass * 1e3:.1f} g "
            f"(ratio {raw.mass / ds_mass:.2f}x)"
        )
        print(f"pinned          : {pinned}")
        print(f"tensor valid    : {pinned.is_physically_valid()}")


if __name__ == "__main__":
    main()
