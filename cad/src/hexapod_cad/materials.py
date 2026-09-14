"""Material densities for mass derivation.

Mass is ``volume * density``, never a guess. Densities are bulk values in
kg/m^3 at room temperature; the comment on each records what it is meant to
represent, because "aluminium" spans 2660-2810 depending on alloy and that
difference is larger than most of the modelling error we care about.

3D-printed materials get an *effective* density that accounts for typical
infill and wall settings, not the filament's solid density. A 20%-infill PLA
part is not 1240 kg/m^3 and pretending otherwise inflates every link mass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """A bulk material with the density used to turn volume into mass."""

    name: str
    density: float
    """kg/m^3."""

    note: str = ""

    def mass_of(self, volume_m3: float) -> float:
        """Mass in kg of ``volume_m3`` cubic metres of this material."""
        if volume_m3 < 0.0:
            raise ValueError(f"volume must be non-negative, got {volume_m3}")
        return volume_m3 * self.density


MATERIALS: dict[str, Material] = {
    # --- metals -------------------------------------------------------------
    "al6061": Material("al6061", 2700.0, "Aluminium 6061-T6 — machined brackets, plates"),
    "al7075": Material("al7075", 2810.0, "Aluminium 7075-T6 — higher strength, same role"),
    "steel": Material("steel", 7850.0, "Mild steel — fasteners, shafts"),
    "stainless304": Material("stainless304", 8000.0, "304 stainless — fasteners, standoffs"),
    "brass": Material("brass", 8500.0, "Brass — threaded inserts, heat-set inserts"),
    # --- 3D printing (effective, infill-aware) ------------------------------
    # Effective = solid_density * (wall + infill fraction). The defaults below
    # assume 3 perimeters at 0.4 mm on a typical 3-6 mm wall part with 20-25%
    # gyroid infill, giving roughly 0.5-0.6 of solid. Override per part if the
    # print profile is known.
    "pla": Material("pla", 700.0, "PLA, ~20% infill effective (solid is ~1240)"),
    "petg": Material("petg", 760.0, "PETG, ~20% infill effective (solid is ~1270)"),
    "abs": Material("abs", 620.0, "ABS, ~20% infill effective (solid is ~1040)"),
    "nylon_cf": Material("nylon_cf", 780.0, "CF-nylon, ~30% infill effective (solid is ~1200)"),
    "pla_solid": Material("pla_solid", 1240.0, "PLA at 100% infill"),
    "petg_solid": Material("petg_solid", 1270.0, "PETG at 100% infill"),
    # --- composites / plate stock -------------------------------------------
    "cf_plate": Material("cf_plate", 1550.0, "Carbon-fibre plate — chassis decks"),
    "fr4": Material("fr4", 1850.0, "FR4 — PCB substrate, bare board"),
    "delrin": Material("delrin", 1410.0, "Acetal/Delrin — bushings, low-friction parts"),
}


def get(name: str) -> Material:
    """Look up a material by key, with a useful error on a typo."""
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(f"unknown material {name!r}; known: {sorted(MATERIALS)}") from None
