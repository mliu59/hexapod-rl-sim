# `cad/` — CAD-driven robot model generation

Builds the hexapod's **realistic** URDF from solid geometry, replacing the
primitive placeholder emitted by `scripts/generate_hexapod_urdf.py`.

Separate division of concern from the RL pipeline: nothing in here imports
Isaac Sim, Isaac Lab, or torch, and nothing in `source/hexapod_lab/` imports
build123d. The interface between them is a generated URDF plus meshes in
`assets/`.

## Why code-CAD

The model is a **Python source file**, not a binary CAD document. That is a
deliberate choice, for three reasons:

1. **No GUI is available.** Development happens over SSH on a headless box.
   Any workflow centred on a CAD render window is unusable here.
2. **Diffs are reviewable.** A geometry change shows up as a source diff, the
   same as every other change in this repo. A `.FCStd` or `.sldprt` is an
   opaque blob.
3. **It is what actually works for agents.** Frontier models reason about
   parametric geometry from *code* far better than from *renders* — on
   BenchCAD, code-QA beats vision-QA by 15–20 points and image-to-code
   collapses to ~0.28 IoU. Feeding an agent numbers and assertions instead of
   pictures is the whole game.

## Stack

| Layer | Choice |
|---|---|
| Kernel | OpenCascade (OCCT), via `cadquery-ocp` |
| Modelling API | [build123d](https://github.com/gumyr/build123d) |
| Agent feedback loop | [build123d-mcp](https://github.com/pzfreo/build123d-mcp) |
| Exchange format | STEP (geometry), STL/OBJ (visual meshes) |
| Output | URDF + meshes into `../assets/` |

`build123d-mcp` is the diagnostics harness, not a convenience. It gives the
agent a persistent CAD session, measurement (volume, area, bbox, topology,
centre of mass), feature detection, preview rendering, and validation. On the
public CADGenBench leaderboard it lifted the *same model* from 0.360 to 0.457
and CAD validity from 88% to 100% — the delta is entirely
build-incrementally-and-measure versus write-a-script-blind.

## Workflow

Order matters, and it is not the human order. **`execute` → `measure` →
`render_view`**, in that sequence: numbers are unambiguous, renders look
correct even when the geometry is wrong. After *every* boolean, check
`topology.faces` — a successful cut changes the face count, a silently failed
one leaves it unchanged and the volume nearly so.

`save_snapshot` before any experiment. Evaluating a "what if we widen this
slot" costs a snapshot, an `execute`, an analysis, and a `restore_snapshot` —
never a hand-redrawn diagram or a speculative source edit.

Two server quirks worth knowing up front:

- **`export()` resolves relative paths against the repo root**, not `cad/`.
  Always pass an explicit path (`cad/out/femur.step`) or it litters the top
  level of the repo.
- **The first VTK-backed call takes longer than the 120 s tool budget** and
  fails. It is a cold-start artifact, not a broken dependency — VTK is ~77 MB
  and loads once per server process. Re-run and it passes. `health_check`
  is the usual casualty since it is often the first thing called.

### Joints, not `.move()`

build123d has native `RigidJoint` / `RevoluteJoint` / `LinearJoint` /
`CylindricalJoint` / `BallJoint`. **Use them for every articulated
relationship.** This is the same idea ArtiCAD calls a "Connector" — a named
local frame with an origin and axis, declared once and then *solved*, rather
than a hand-computed transform. Three reasons it matters here:

1. `RevoluteJoint` is the direct analog of a URDF `revolute` joint, axis and
   limits included. 18 of them is the robot.
2. Move a parent and children follow. With raw `.move()` the relationship is
   gone and every downstream offset silently goes stale.
3. It keeps spatial reasoning out of the loop — the weakest link. Frames get
   declared, the solver composes them.

### Batteries already included

`version()` reports these as importable inside `execute()`:

- **`bd_warehouse`** — threads, fasteners, gears, bearings. M2.5 hardware and
  bearings do *not* need vendor STEP; `ClearanceHole`/`TapHole`/
  `CounterSinkHole` take a fastener object and compute head geometry and
  tap-drill diameters themselves. Never hand-roll these.
- **`augura`** — printability analysis, which is how the printed-vs-machined
  question gets answered with numbers instead of opinion.

### Visuals

Development is over SSH with no display, so rendered output is published as an
Artifact rather than written to `out/` and forgotten. For design-discussion
diagrams use the `build123d://presentation` cookbook — note it must run from a
script *outside* the MCP sandbox, which blocks `ExportSVG.write()`. Multi-view
dimensioned drawings have moved out of this server to `draftwright`; the
server's own drawing tools are deprecated (off by default from 0.4.0, removed
in 0.5.0).

Scale `Draft` parameters to the part. Defaults are tuned for A4, and on a
25 mm part the default `line_width=0.5` and `arrow_length=3.0` render witness
lines as thick filled rectangles. Override all of them, not just `font_size`.

## Inertia policy

**Inertias are never guessed and never assigned by a language model.** Every
link's mass, centre of mass, and inertia tensor is computed from real geometry.
Two sources, both first-class:

- **Authored solids** — structural parts modelled in build123d. Mass is
  `volume × density` for a declared material; the inertia tensor comes from
  OCCT's `matrix_of_inertia` on the solid.
- **Vendor parts** — servos, brackets, bearings, fasteners. Real STEP from the
  manufacturer where available; mass from the datasheet where the STEP is a
  simplified envelope, since vendor CAD is often hollow or decorative.

A link is usually a *compound* of both (e.g. a femur bracket plus the servo it
carries), so link inertia is the composed rigid-body sum about the link frame,
not any single part's tensor.

Collision geometry stays **primitives** (box / capsule / sphere / cylinder)
fitted to the solids — never convex decomposition of the visual mesh. This is a
hard Isaac Lab requirement for 4096-env throughput, restated in `INTRO.md`
Phase 0.

## Environment

Isolated from the Isaac stack on purpose — see `CLAUDE.md` for why the repo
`.venv` must not receive heavy installs.

```bash
cd cad
uv sync                      # creates cad/.venv on Python 3.12
uv run python -c "import build123d; print(build123d.__version__)"
```

The MCP server is registered in the repo's `.mcp.json` and runs out of its own
uv tool environment; it does not use `cad/.venv`.

## Layout

```
cad/
  src/hexapod_cad/     importable package (materials, mass properties, parts)
  vendor/              vendor STEP files, one directory per part, with PROVENANCE
  out/                 generated artifacts — gitignored, reproducible
  pyproject.toml       uv project
```

## URDF pipeline — how robot iterations become URDFs

```bash
cd cad && uv run python scripts/build_robot.py
# -> assets/urdf/spidertron.urdf + assets/meshes/spidertron/{body,coxa,femur,tibia}.stl
```

Emission, not conversion: `src/hexapod_cad/params.py` is the single source of
truth (geometry, joint limits, servo specs, masses, naming); `links.py`
builds each link's solids in its URDF link frame; `urdf.py` writes joints
from the same constants, per-link STL visuals (shared across all six legs),
primitive collisions, and exact `massprops` inertials. Validation is part of
the build (yourdfpy parse, FK feet-on-ground, mass budget) — a failing build
exits nonzero. **To iterate the robot: edit `params.py`/`links.py`, rerun.
Never edit the generated URDF.** Design rationale and iteration history:
[DESIGN_LOG.md](DESIGN_LOG.md).

## Status

Spidertron-style 18-DOF hexapod designed, collision-verified, and emitting
`assets/urdf/spidertron.urdf`. `materials.py` and `massprops.py` are the
verified spine of the inertia policy; vendor servo CAD lives under
`vendor/` with provenance.
