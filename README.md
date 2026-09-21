# Bolt Group Optimizer

A local web tool that lays out a bolt pattern on a 2D DXF part so the peak
per-bolt shear load (elastic/polar method) stays under a target ceiling,
using as few bolts as possible. See `CLAUDE.md` for the underlying physics
and optimization approach.

## Setup

Python 3.11+. Any environment with `torch` works; a fresh venv is simplest:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt
```

`requirements.txt` does not pin a torch build so you can pick CPU or CUDA.
If you already have a conda env with torch, just `pip install -r requirements.txt` there.

## Compute backends

The optimizer can run on three backends, chosen at run time from the **Compute
backend** dropdown in step 5. Only the ones this machine can actually use are
offered — `GET /device` probes each and returns `{available, default, details}`,
and the adapter / GPU name is shown beside the dropdown.

| backend | device | precision | needs |
| --- | --- | --- | --- |
| `cpu` | CPU | float64 | nothing — always available, always the default |
| `cuda` | NVIDIA GPU | float64 | a CUDA-enabled `torch` build |
| `directml` | any DirectX 12 GPU (AMD, Intel, NVIDIA) | **float32** | `torch_directml`, in its own venv |

Precision is a property of the backend, not a setting: **DirectML has no
float64 kernels at all**, so the search runs in float32 there. The final
scoring pass is always redone on the CPU in float64 whatever the search ran on,
so the loads, `lc`, `k` and feasibility flags you read and export are identical
across backends — only the search path differs. The results header names the
backend that actually ran (a request for one that is not present quietly
degrades to `cpu`) and says whether the DirectML op substitutes were in play.

### Install 1 — standard (CPU / CUDA)

The Setup section above. Install a CUDA torch build if you want the `cuda`
backend.

### Install 2 — DirectML (separate venv, required)

`torch_directml` **hard-pins the torch it was built against** — 0.2.5.dev240914
pins `torch==2.4.1` — so it cannot share an environment with a newer torch
(ToastEnv has 2.10). It is also published only as pre-releases, hence `--pre`.
Give it its own venv:

```powershell
py -3.11 -m venv .venv-dml
.venv-dml\Scripts\python.exe -m pip install --pre -r requirements-directml.txt
```

`.venv-dml/` is gitignored. `run.py`, the test suite and the tools work
unchanged from either environment — the only difference is which backends
`GET /device` reports.

```powershell
.venv-dml\Scripts\python.exe run.py     # server with cpu + directml
```

Two helper scripts live in `tools/`:

* `tools/dml_probe.py` — runs every torch op the optimizer uses on the DirectML
  device and prints an OK / FAIL / **CPU-FB** table. `CPU-FB` is the row to
  care about: `torch_directml` does not raise on an op it lacks a kernel for,
  it silently copies the call to the CPU and back.
* `tools/bench_backends.py` — the same optimizer run on several backends from
  one identical seed, for comparing N / peak / wall time.

DirectML is not automatically faster. This problem is small (a few thousand
2-vectors) and DirectML's per-kernel launch overhead is high, so at modest seed
counts the CPU wins outright; see `tools/bench_backends.py` for your own
machine's crossover.

Set `BOLTOPT_FORCE_FALLBACKS=1` to exercise the DirectML substitute code paths
on an ordinary CPU run — the test suite uses this so they stay covered.

## Run

```powershell
python run.py
```

This starts the server on `http://127.0.0.1:8000` and opens it in your browser. (Equivalent to `uvicorn app.main:app --port 8000`.)

## Usage (click workflow)

The sidebar is organised as seven numbered, collapsible steps. A status line
at the top lists what is still missing before the optimizer can run ("Select
the part outline", "Pick a datum", …) and the Run button stays disabled, with
the same reasons in its tooltip, until everything is in place. Errors and
warnings appear as toasts over the canvas and clear themselves; a spinner
("Parsing DXF…", "Building geometry fields…") shows whenever the backend is
busy for more than a moment. **Esc** leaves the current click mode; **Delete**
removes the fixed bolt under the cursor. Left-drag on empty canvas pans, the
wheel zooms, middle-drag / Space+drag also pan.

1. **Load** — upload a DXF. Closed loops are drawn thin gray; loops that
   failed to chain into a closed boundary are drawn red (open chains). The
   step shows the effective units (from `$INSUNITS`, or the override) and
   notes anything the loader skipped (unsupported entity types, geometry that
   only exists in a paper-space layout). Loading a new file clears any
   previous result.
2. **Part & keepouts** — click "Select part", then click the loop that is the
   part's outer boundary. It highlights blue; click it again to clear. Every
   other closed loop's role decides what it means:
   * **Unmarked** — a hole: removed material, forbidden to bolts, and a real
     bearing/tear-out edge — with one exception: a hole that has a fixed bolt
     placed in it (step 4) counts as occupied, so the bolt is legal there and
     bears against the plate around the hole rather than against its own hole
     wall.
   * **Keepout** ("Select keepout", orange fill) — solid material: forbidden
     to bolts (subtracted from the legal placement region, with the same
     inward buffer as the outer edge), but *not* a bearing edge, since it is
     never subtracted from the material region. Use this for e.g. a
     bolt-circle reference circle that should block placement without being
     misread as a hole.
   * **Ignore** ("Mark ignore", faint dashed) — construction geometry with no
     geometric effect at all: excluded from both regions, drawn only for
     context.

   "Show legal region" overlays the placement region (green, dashed) for the
   current clearance, keepouts and holes, and follows settings changes; the
   part highlight fill always includes keepout and ignore areas as solid
   material, with the keepout's own outline drawn on top of it.
3. **Datum & forces** — click "Pick datum", then click near a DXF vertex or
   circle/arc center (snap markers appear as small yellow dots in pick modes;
   the nearest one within ~10px highlights). The datum is drawn as a cyan
   crosshair with short x/y axis stubs; every reported coordinate is relative
   to it (+x right, +y up). Force cases live in the same step (see 6 below).
4. **Fixed bolts** (optional) — click "Place bolt", then click anywhere in the
   part for each bolt you want to position yourself. The click snaps to the
   nearest snap point (hole centre / vertex) if one is within ~10px, otherwise
   it lands exactly where you clicked. Placed bolts draw as a hollow green
   circle with a cross through it, at the real bolt diameter, and are listed
   in the sidebar with datum-relative coordinates and a remove button ("Clear
   all" wipes the list). On the canvas every bolt in the displayed layout is
   numbered by its layout index (1, 2, 3 …), matching the B1, B2 … rows of the
   per-bolt table; a fixed bolt placed after a run joins the displayed layout
   immediately and the numbers follow it.

   A fixed bolt is a **full member of the bolt group**: it carries load, counts
   toward the centroid and polar moment, and takes part in the spacing and
   bearing/tear-out checks — the optimizer simply never moves it, and places the
   remaining bolts around it. "N min"/"N max" still count *total* bolts, so a
   sweep starting below the number of fixed bolts is clamped upward and the
   result message says so.

   Each bolt is validated against the current legal placement region when it is
   placed, and again when you change the edge clearance. A bolt stranded outside
   the region (too close to an edge after a larger clearance, or inside a
   keepout) is drawn red, highlighted in the list, and named in the error line;
   move it, remove it, or loosen the clearance before running.
6. **Force cases** (in step 3) — click "+ Add case" for each independent load
   case (e.g. one for the mechanism extended, one for it closed); adding a
   case drops you straight into picking its point. Each row has an
   editable name, a magnitude, an angle (degrees, 0 = +X, CCW positive) and a
   "Pick point" button that re-enters the snap-click workflow for *that* case.
   Every case's arrow is drawn on the canvas with a "name  magnitude @ angle"
   label placed clear of the bolt markers; the case currently being picked is
   highlighted. Cases are evaluated **independently** on the
   same bolt group — never superposed — and the optimizer minimizes the peak
   per-bolt load over all (case, bolt) pairs, so the layout has to clear the
   ceiling in every case. Results show the peak for each case and mark the
   governing one; the per-N table keeps showing the overall peak.
7. **Settings** (step 5) — edge clearance, min bolt spacing, **bolt
   diameter**, a **Bearing model** toggle, load ceiling, N sweep range
   (N min is at least 1), **seeds per N**, iteration budget and the **Compute
   backend** dropdown (see "Compute backends" above — it lists only the
   backends this machine can actually run, with the GPU / DirectML adapter
   name beneath it). The units override lives in step 1. Each length
   field shows the effective unit next to it.

   **Results always reflect the current inputs.** Changing any setting, force
   case, datum or the part/keepout selection while a layout is on screen
   re-scores that layout (same positions, new loads / `lc` / `k` /
   feasibility) and the tables update; `/result` and the DXF export do the
   same on the server, so nothing stale can be read or exported. The per-N
   summary and alternatives list stay as the run reported them, with the
   ceiling ticks recomputed against the current ceiling.

   **Bolt diameter / Bearing model** control the bearing / tear-out model
   (see "Bearing / tear-out model" in `CLAUDE.md`). With it on — the default —
   the optimizer no longer treats "sitting on the edge-clearance boundary" as
   free: each bolt's capacity is scaled by the clear distance `lc` from its hole
   edge to the nearest material edge (outer profile, hole/keepout, or an
   adjacent bolt hole) *in the direction that bolt pushes on the plate*,
   ramping up to full capacity at `lc = 2d`. That direction is the **opposite**
   of the bolt's load resultant — a plate pulled left tears out at its
   right-hand end — so it is the edge *behind* the bolt relative to the pull
   that matters. The load reported and compared against the ceiling is
   the **effective** load `raw / k`, so a crowded bolt reads worse and the
   optimizer moves it away from the edge it bears against. Edge clearance still
   applies as a hard floor. Turning the toggle off restores the plain elastic
   peak. The canvas draws bolts at their real diameter and puts a short yellow
   tick on each one showing its **bearing direction** in the governing case
   (labelled in the legend, bottom-left of the canvas).

   "Seeds per N" replaces the old "Restarts": the optimizer no longer sweeps
   N one value at a time. It generates `seeds_per_n` diverse starting layouts
   for *every* N in the sweep range (so 80 x N=2..12 is ~880 layouts) and
   optimizes all of them simultaneously in one batched torch run. More seeds
   buys better coverage of the local minima at a roughly linear cost; the
   default of 80 finishes in a few seconds on CPU.
8. **Run** (step 6) — starts the optimizer over a WebSocket. The geometry
   fields are built first ("Building geometry fields…" spinner; they are
   cached and shared with the drag-evaluate loop, so a run never rasterizes
   twice), then the current best layout animates live with a progress bar
   and iteration / surviving batch size / best N + peak. When it finishes,
   step 7 **Results** shows a header with the key numbers (N, effective peak,
   ceiling, margin %, governing case, and an "under ceiling" / "over ceiling"
   / "infeasible" badge), then compact tables: **per case** (effective and raw
   peak, worst bolt, governing row highlighted), **per bolt** (datum-relative
   coordinates, raw load, `lc`, `k`, effective load and the limiter — outer
   edge / hole / adjacent bolt, shown as "—" once `lc ≥ 2d` and no limiter
   applies; the governing bolt's row is highlighted and it wears a dashed
   orange ring on the canvas), the **force cases** (point, vector, magnitude,
   angle), a **per-N summary** listing the best achievable peak and margin at
   every bolt count in the sweep, so you can see directly whether e.g. N=3
   was even close, and an **Alternatives** list of up to five geometrically
   distinct layouts at the winning N — click one to display it, and both the
   tables and "Export DXF" follow whichever is currently displayed. Fixed
   bolts keep their distinct marker and are ticked in the per-bolt table.
9. **Drag bolts** — after a run (or with only fixed bolts placed) every bolt on
   the canvas is draggable, in any mode: hover one and the cursor turns into a
   grab hand, and dragging it takes precedence over that mode's click action and
   over left-button panning for the duration of the gesture (middle-mouse and
   space-pan are unchanged). While you drag, the layout is re-scored live by
   `POST /evaluate` — the *same* code path as the optimizer's own final scoring,
   so the numbers agree exactly — at about 30 Hz: the per-bolt table (load, `lc`,
   `k`, eff, limiter, fixed), the per-case peaks and governing case, the header
   peak and the bearing ticks all follow the bolt, and a bolt turns **red** when
   its effective load is over the ceiling, when it lands outside the legal region
   (a keepout, a hole, or inside the edge clearance) or when it is closer than
   the minimum spacing to a neighbour. The header then shows a red *infeasible*
   tag and names the offending bolts. Dropping a bolt outside the region is
   allowed — nothing snaps back — but **Export DXF then refuses** with that same
   message, so an illegal layout can be looked at but not shipped.

   Dropping a bolt makes the edited layout the current one ("Manual layout
   (edited from optimized result)"), and two buttons appear:

   * **Revert** — puts the layout it was edited from back, exactly (bolt
     coordinates unchanged to the bit), including the pinned-bolt list.
   * **Re-optimize around this** — re-runs the optimizer at the *same* bolt
     count with the bolts you actually dragged (plus any already-pinned ones)
     held where you put them and the rest free. It pins them by adding them to
     the fixed-bolt list and setting N min/max to the current count, so you can
     see in the sidebar exactly what it ran. "Stop" works as usual.

   Fixed bolts (F1, F2 …) are draggable too; dropping one also updates the
   pinned list, and is rejected there with the usual named error if it landed
   outside the region.

   "Export DXF" returns the original DXF with
   added BOLTS/FIXED_BOLTS/DATUM/FORCE layers (bolt circles at their real
   diameter — user-placed ones on the FIXED_BOLTS layer with a cross through the
   hole — one labelled arrow per force case, and the datum as a crosshair with
   labelled +X / +Y stubs) and a note block listing the datum, units, all
   coordinates relative to the datum, per-case peaks, a "(fixed)" tag on each
   user-placed bolt and, with bearing on, each bolt's `lc` / `k` / effective
   load plus the bearing assumptions (`d`, saturation at `2d`). The note is
   MTEXT on R2000+ drawings and stacked TEXT lines on an R12 source (which
   has no MTEXT).

## Tests

```powershell
conda activate ToastEnv
python -m pytest tests/ -v
```

## Sample files

`samples/make_samples.py` generates three fixture DXFs (already committed
under `samples/`):

1. `plate_with_holes.dxf` — rectangular plate + two round holes.
2. `l_bracket.dxf` — L-bracket drawn as loose LINEs with 0.001-unit gaps at
   two joints (tests loop chaining), plus a circle, a small keepout
   rectangle, and one stray unconnected line.
3. `plate_open_loop.dxf` — a plate with an interior loop that has an
   intentional 5-unit gap, so it stays flagged as an open chain instead of
   closing.

Regenerate with:

```powershell
conda activate ToastEnv
python samples\make_samples.py
```
