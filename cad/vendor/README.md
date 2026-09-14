# Vendor part CAD

Manufacturer STEP files for off-the-shelf components — servos, brackets,
bearings, fasteners. One directory per part.

Each part directory must contain a `PROVENANCE.md` recording:

- **Source URL** and the date downloaded
- **Manufacturer part number** (exact — `FR12-H101K` is not `FR12-S101K`)
- **Datasheet mass** in grams, and the URL it came from
- **Geometry fidelity** — is the STEP a true solid, a simplified envelope, or a
  shell? This determines whether `of_step()` gets a `datasheet_mass` override.
- **Licence / redistribution terms**, since some vendors restrict CAD reuse.

The provenance file is not bureaucracy. Vendor CAD is routinely a decorative
envelope with no internals: integrating it at the material density can be off
by 2-3x in either direction, and nothing downstream will flag it. The
datasheet mass is the correction, and six months from now the only way to know
which parts were corrected — and why — is if it was written down at import
time.

`hexapod_cad.massprops.of_step(path, material, datasheet_mass=...)` keeps the
mass *distribution* from the geometry while pinning the *magnitude* to the
datasheet. Prefer that over trusting raw volume for any part you did not model
yourself.

## Parts in this directory

- `sts3215/` — Feetech STS3215 (coxa yaw ×6). Official manufacturer STEP.
- `sts3250/` — Feetech STS3250 (femur + knee pitch ×12). Official
  manufacturer STEP.

Both files integrate to a fraction of their datasheet mass (0.50× and
0.22× respectively — shelled internals), which is exactly why the
`datasheet_mass` override exists. See each PROVENANCE.md.

M2.5 hardware and bearings come from `bd_warehouse`, not vendor STEP.
