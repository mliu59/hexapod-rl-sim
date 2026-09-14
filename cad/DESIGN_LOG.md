# Design log — spidertron-style hexapod, v1 concept

How the design was reached, including the dead ends. Companion to
`analysis/static_torque.py` (the sizing math) and the published design
review artifact. Date: 2026-09-12.

## Brief

Spidertron-inspired hexapod (the Factorio octopod, six-legged variant):
compact pod body, long spindly legs, high knees. Hobby build assumptions —
PLA on an FDM printer, light chassis load. Hard requirement: **walk over
200 mm obstacles**. Actuators must be real, purchasable servos, selected by
analysis rather than vibes.

## Order of operations

Requirement → geometry → mass budget → torque analysis → actuator selection →
CAD → joint-range collision verification → renders. Each step feeds the next;
the two iterations below happened where a later step falsified an earlier
guess.

## Geometry

The 200 mm obstacle sets a floor on body clearance and foot lift. Chosen:

| Parameter | Value |
|---|---|
| Coxa (yaw axis → femur axis) | 60 mm |
| Femur | 190 mm |
| Tibia | 380 mm |
| Hip axis height (standing) | 270 mm |
| Body underside clearance | 230 mm |
| Standing pose | femur +30°, knee interior 76° |
| Stance footprint | ~850 mm across |

The spidertron high-knee posture (femur up, knee above the body line, long
tibia down) is not just styling: it converts leg length into swing-foot
clearance while keeping the loaded moment arm bounded. Foot lift capability
at femur +85°/knee 60°: foot at 241 mm; verified in CAD with an actual
200 mm block — 41.3 mm clearance over it, body 200.9 mm clear.

## Iteration 1 — the torque sweep vetoed the first envelope

An unconstrained stance sweep (any reach, any crouch) put the sizing pose at
a fully-splayed crouch: reach 340 mm at hip 200 mm → **49.4 kg·cm** femur —
just past the biggest affordable servo's stall. That pose serves no function,
so it was excluded *by design*: stance reach ≤ 280 mm, crouch floor
hip ≥ 210 mm. These envelope numbers are load-bearing and must become URDF
joint limits + gait constraints. Inside the envelope the worst single-leg
case (W/2, rough-terrain transient) is 40.7 kg·cm femur / 24.7 coxa /
21.8 knee.

## Actuator selection

Mass budget 2.90 kg (18 servos = 1.25 kg of it). Candidates researched:
Feetech STS3215 (30 kg·cm @12 V, 60 g, ~$16), STS3250 (50 kg·cm, 74.5 g,
~$40), Dynamixel XL430-W250 (15.3 kg·cm, $50), XM430-W350 (41.8 kg·cm, $270).

**Selected: STS3250 ×12 (femur + knee), STS3215 ×6 (coxa), ~$576 total.**
XL430 fails on torque; XM430 fails on price (~$4.9k for 18). The two Feetech
cases share an identical 45.2×24.7×35 mm footprint → one bracket design, one
12 V TTL bus.

Safety factors: 1.2× stall on the worst transient (coxa and femur), 2.3× at
the knee. Honest weakness: sustained *walking* femur load (26.1 kg·cm at
W/3) is ~2× the 13 kg·cm rated figure — the servos will run warm, like every
hobby hexapod. Standing on six legs sits exactly at rated (13.1). The sim
enforces the derated envelope anyway: effort limits 3.43 N·m (pitch) /
2.06 N·m (yaw), velocity ~5.4–5.9 rad/s.

## Iteration 2 — CAD falsified three "finished" details

Building the geometry as solids and checking it at joint extremes found
problems the analysis could not see:

1. **Coxa yaw range.** ±35° felt reasonable; at full adjacent convergence the
   legs interpenetrate (1,169 mm³ overlap — later 88.6 mm³ with the final
   slimmer parts, still a hard collision). URDF limits are unconditional, so
   the range was cut to **±25°**, giving 47.6 mm worst-case clearance.
2. **Femur plates were flat, not vertical.** A `SlotOverall` sketch extruded
   in the wrong plane made the "twin plates" 30 mm wide horizontally,
   blocking the tibia's swing plane entirely. Rebuilt vertical.
3. **The knee joint had three mutually-fighting parts.** The knee servo case
   initially extended *past* the joint into tibia space; the round shin
   (r=11, centred on the joint plane) fought the case, horn, and plates for
   the same 16.9 mm channel at every fold angle. Fix: tuck the case back
   along the femur (mirrored mount) and make the tibia a single-side tapered
   arm riding the horn face, confined to y ∈ [0, 12] — clear of the case
   (≤ −3.5), the horn (≤ 0 at the hub only), and the plates (≥ 13.35) at
   every knee angle from 35° to 150°.

Also fixed en route: build123d `Cone` is origin-centred (first tibia stuck
backward through the knee — caught because the feet floated 9 mm off the
ground); hip servo bays had to be carved into the pod so the mounts are real
recesses rather than documented interpenetration; and `Compound(children=…)`
*re-parents* shared children (assembling a pose-study scene silently stole
the body out of the main assembly — caught by a face-count drop from 413 to
378, exactly the body's 35).

## Tooling lessons (for the next session)

- **Never trust `&` or `clearance()` on nested Compounds.** Boolean common on
  them returned 295,665 mm³ for two legs 700 mm apart — larger than one leg's
  entire volume. Fuse to single solids first; calibrate against known-disjoint
  and known-overlap boxes when a number looks wrong.
- The verification matrix on fused solids is cheap (~seconds/case). Sweep the
  full knee range, not just endpoints — two of the collisions found were
  angle-independent (assembly fit), which endpoint checks would have
  misattributed.
- Constant-across-angles interpenetration = assembly-fit problem at the
  joint, not a swing collision. Different fix category.

## Verified end state

| Check (at joint-range extremes) | Result |
|---|---|
| Adjacent legs, both yaw 25° converging | 0 overlap, 47.6 mm clear |
| Swing leg over stance neighbour, yaw limits | 0 overlap, 47.6 mm clear |
| Yaw ±35° (rejected range) | collides — evidence for ±25° |
| Max lift (pitch +85°, knee 35°) vs body | 0 overlap, 10.0 mm clear |
| Reach-down (pitch −25°, knee 150°) vs body | 0 overlap, 10.0 mm clear |
| Knee fold sweep 35°–150°, tibia vs femur+servo | 0 overlap at all angles |
| Step over 200 mm block (pitch 85°, knee 60°) | foot 232 mm, 41.3 mm clear |
| Validity gate (7 solids: body + 6 fused legs) | PASS, watertight, 0 overlaps |

Export: `cad/out/spidertron_hexapod.step` (regenerable; out/ is gitignored).

## Joint ranges (v1, feed into URDF)

| Joint | Range | Standing | Limited by |
|---|---|---|---|
| Coxa yaw | ±25° | 0° | adjacent-leg collision (was ±35°) |
| Femur pitch | −25° … +85° | +30° | body bay clearance / reach-down use |
| Knee interior | 35° … 150° | 76° | assembly fit verified across full range |

## Vendor CAD integration (2026-09-12, follow-up)

Official Feetech STEP files replaced the parametric servo blocks:

- Located via the manufacturer's reseller channel (feetechrc.com's own
  download page 404s): `ST3215-20200328-to-customer.stp` and
  `st-3250m-20211119-B.stp`, both authored in Pro/ENGINEER with Feetech
  drawing-numbered filenames — genuine manufacturer exports, not community
  remodels. Stored under `vendor/sts3215/` and `vendor/sts3250/` with
  PROVENANCE.md each.
- **Both files are mass-dishonest, as predicted**: integrated volume ×
  density gives 0.50× (3215) and 0.22× (3250) of the datasheet mass — the
  gearbox internals are shelled. `of_step(..., datasheet_mass=)` pinning is
  mandatory for these; both pinned tensors pass `is_physically_valid()`.
- Frame decoding: dual-shaft servo, output axis vertical at 10.1 mm from the
  case rear, horn top proud at y=9.2, idler hub on the bottom face. A
  fit-hull (case shells + horn + idler, internals dropped: 851/924 faces)
  was remapped to the leg convention (shaft +Z, horn interface z=0).
- The knee-fold sweep re-ran clean against the *real* horn/case geometry —
  the single-side tibia arm clears at every angle, same as with the blocks.
- Discovered mass discrepancy: Feetech's own spec says the STS3215 is
  55 ± 1 g, not the 60 g used in `static_torque.py` (from servodatabase).
  Kept 60 g — conservative, and worth 30 g total across six coxae.
- Discovered position discrepancy: the real output shaft is 10.1 mm from the
  case rear, not the 11.3 mm the parametric block assumed from the datasheet
  drawing. Corrected in the working model.

### Iteration 3 — the vendor detail had to come back out

Instancing the ~900-face hulls 18× made the assembly 16,397 faces.
Consequences: renders went from seconds to timing out entirely (tessellation
is face-count × curvature bound), and exact booleans/distance checks blew the
120 s tool ceiling repeatedly. Resolution — the standard two-representation
split, same shape a URDF wants:

- **Ground truth**: official STEP in `vendor/`, used for mass properties
  (datasheet-pinned) and detail reference. Rendered rarely.
- **Working model**: `servo_lp()`, a 10-face envelope with the exact outer
  dimensions and true shaft position measured from the official geometry.
  Verified to *contain* the real hull (protrusions: 0.5–34 mm³ of sub-mm
  seam slivers) — so clearances measured on it are conservative lower
  bounds. Assembly back to 395 faces; renders instant at high quality.

One more kernel-trust lesson: widening the envelope box by 0.02 mm flipped
`hull.cut(box)` from 1.5 mm³ to 11,242 mm³ — nonsense (a larger cutter
cannot leave more material). OCCT booleans near coincident faces are
unstable; keep deliberate clearances rather than exact coincidence, and
sanity-check any boolean result against monotonicity expectations.

## URDF pipeline (2026-09-12, follow-up)

Built as emission from the parametric source, not conversion from geometry —
every CAD→URDF converter exists to recover joint frames and inertias from an
opaque document, a problem code-CAD doesn't have. `params.py` (single source
of truth) → `links.py` (per-link solids in URDF link frames) → `urdf.py`
(joints/limits/inertials/meshes) → `scripts/build_robot.py` (rerunnable,
self-validating). Choices:

- **Per-link STL visuals, one file per link type** — all six legs share
  geometry (radial symmetry, no mirroring), so 4 STLs cover 19 links.
- **Primitive collisions from parameters**, cylinders emitted so Isaac's
  `replace_cylinders_with_capsules` upgrades them at import.
- **Stator-side mass assignment**: yaw servo case → body link, hip pitch
  case → coxa, knee case → femur (corrects the session mockup, which drew
  the hip case rotating with the femur).
- **Sign conventions match the legacy placeholder** (femur pitch about +y,
  negative = lift; knee positive = flexion) so `_DEFAULT_JOINT_POS` and the
  actuator regexes port unchanged. Standing pose: coxa 0, femur −0.5236,
  tibia +1.8151 rad.
- **Validation inside the build**: yourdfpy parse + kinematic tree, FK on
  the *emitted* URDF putting all six feet within 3 mm of ground, per-link
  tensors `is_physically_valid()`, total mass within the torque-analysis
  budget window. First full run: 960/104.5/134.5/85 g links, 2904 g total —
  exactly the budget.

## Deliberately deferred

Printable part detailing (wall thickness, bolt bosses, print orientation),
per-link mass properties via `hexapod_cad.massprops` (compose printed part +
servo datasheet mass), URDF emission from this geometry, `augura`
printability pass, real vendor STEP for the STS servos (currently
parametric blocks at correct external dimensions and datasheet mass).
