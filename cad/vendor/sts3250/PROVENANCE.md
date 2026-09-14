# Feetech STS3250 — vendor CAD provenance

- **File**: `STS3250.step` (10.4 MB)
- **Source URL**: https://cdn.shopify.com/s/files/1/0673/6848/5000/files/st-3250m-20211119-B.stp?v=1761662822
  (AIFITLAB product page https://aifitlab.com/products/feetech-sts3250-servo-motor,
  a Feetech reseller mirroring the manufacturer's drawing `st-3250m-20211119-B`)
- **Downloaded**: 2026-09-12
- **Origin evidence**: Feetech drawing-numbered filename; Pro/ENGINEER STEP
  export consistent with the STS3215 companion file.
- **Part number**: STS3250 (12 V, 50 kg·cm)
- **Datasheet mass**: 74.5 ± 1 g (AIFITLAB / servodatabase agree)
- **Datasheet dims**: 45.22 × 24.72 × 35.0 mm (case, excl. horn/lugs)

## Geometry fidelity (from `analysis/inspect_vendor_step.py`)

13 solids; bbox 45.22 × 37.40 × 24.72 mm — matches datasheet with the horn
on the 37.4 axis. Integrated volume just 11.9 cm³ → raw volume × density is
~0.22× the datasheet mass: **heavily shelled, metal gearbox not modelled as
solid metal**. Always use `of_step(path, ..., datasheet_mass=0.0745)`.
Pinned inertia tensor passes `is_physically_valid()`.

- **Licence**: distributed by the manufacturer to customers for design-in
  use; no explicit licence text. Do not redistribute outside this repo.
