# Feetech STS3215 — vendor CAD provenance

- **File**: `STS3215.step` (5.7 MB)
- **Source URL**: https://cdn.shopify.com/s/files/1/0673/6848/5000/files/ST3215-20200328-to-customer.stp?v=1761662975
  (AIFITLAB product page https://aifitlab.com/products/feetech-sts3215-servo-motor,
  a Feetech reseller mirroring the manufacturer's customer file)
- **Downloaded**: 2026-09-12
- **Origin evidence**: STEP header `FILE_NAME('ST3215-20200328_ASM', '2020-03-28T', ('WUSALT01'), ...)`,
  authored in Pro/ENGINEER — Feetech's own engineering export ("to-customer"
  in the filename), not a community remodel.
- **Part number**: STS3215 (12 V, 30 kg·cm variant used in this project;
  the mechanical envelope is shared across STS3215 voltage/gear variants)
- **Datasheet mass**: 55 ± 1 g (AIFITLAB spec table; note servodatabase lists
  60.0 g — `analysis/static_torque.py` uses 60 g, conservative)
- **Datasheet dims**: 45.2 × 24.7 × 35.0 mm (case, excl. horn/lugs)

## Geometry fidelity (from `analysis/inspect_vendor_step.py`)

16 solids; bbox 45.23 × 37.40 × 24.73 mm — matches datasheet with the horn
on the 37.4 axis. Integrated volume 19.6 cm³ → volume × density gives only
~0.50× the datasheet mass: **simplified internals, do not trust raw
volume**. Use `of_step(path, ..., datasheet_mass=0.055)` — keeps the mass
distribution from the geometry, pins the magnitude to the datasheet.
Pinned inertia tensor passes `is_physically_valid()`.

- **Licence**: distributed by the manufacturer to customers for design-in
  use; no explicit licence text. Do not redistribute outside this repo.
