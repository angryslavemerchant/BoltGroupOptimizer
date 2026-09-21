# Bolt Group Optimization — Spec

## Goal

Given a 2D part shape, an applied force (magnitude, direction, and point of application), and placement rules, find a bolt pattern — count and positions — that keeps the peak per-bolt load under a target ceiling, using as few bolts as reasonably possible. The core insight: bolt load for a given layout is a closed-form formula (the elastic/polar method), and that formula is differentiable with respect to bolt position, so the layout can be found by gradient-based optimization instead of manual iteration.

## Inputs

- **Shape geometry** — the part's 2D boundary (outer profile)
- **Force cases** — one or more *independent* load cases, each its own point of application plus magnitude and direction (a case fully determines both the direct shear and the moment about any bolt group's centroid); one layout must be acceptable under all of them, and cases are never superposed
- **Keepout zones** — one or more regions (interior or boundary-adjacent) where no bolt may be placed. A keepout is *solid material*, not a hole: it is subtracted from the legal placement region only, never from the material region, so it carries no bearing/tear-out edge (e.g. a bolt-circle reference circle marked keepout blocks placement without being misread as a hole). A loop marked **ignore** is pure construction geometry: it is excluded from both regions entirely and has no geometric effect.
- **Edge clearance** — minimum distance any bolt center must keep from the part boundary
- **Bolt-to-bolt spacing** — minimum distance between any two chosen bolts
- **Load ceiling** — the maximum allowable per-bolt shear force
- **Fixed bolts** (optional) — bolts the user places by hand, typically on existing hole centres

**Fixed bolts.** Any bolt the user places by hand is a *full member* of the bolt
group that simply never moves: it carries load, counts toward the group centroid
and polar moment, participates in the minimum-spacing and bearing/tear-out
checks, and appears in the results table and the DXF export like any other bolt.
The optimizer places the remaining bolts around it. Implementation-wise the `F`
fixed bolts occupy slots `0..F-1` of every layout in the batch, live and pinned —
their gradient is zeroed and their coordinates are rewritten after each Adam
step, and they are excluded from seeding and from the final shapely snap. `N`
still counts *total* bolts, so the sweep is clamped to `N ≥ F`. Free seed slots
are drawn from a pool with everything within `min_spacing` of a fixed bolt
removed, so no layout starts out violating spacing against a bolt that cannot
move out of the way. A fixed bolt must lie inside the legal placement region;
raising the edge clearance under one is reported as a named error rather than
silently moving it.

## Geometry layer (candidate region)

Before any optimization runs, derive the *legal placement region*:

1. Take the part boundary and subtract every keepout zone
2. Shrink (offset inward) what remains by the edge-clearance minimum
3. The result is the region any bolt center is allowed to sit in

This step is pure computational geometry (polygon boolean + offset operations) — no physics, no optimization. It only needs to run once per part configuration, and its output either bounds where continuous bolt coordinates can move (as a hard constraint) or seeds a discrete grid of candidate points, depending on which optimization strategy is used (see below).

## Objective — peak bolt load

For a given set of bolt positions, the elastic (polar-moment) method gives per-bolt shear load in closed form:

1. Compute the centroid of the bolt group
2. **Direct shear**: applied force split evenly across all bolts
3. **Moment shear**: the applied force's moment about the centroid, distributed to each bolt in proportion to its distance from the centroid, acting perpendicular to the line from centroid to that bolt
4. Total per-bolt load = vector sum of direct + moment components
5. Objective = the maximum of these per-bolt loads (the worst-loaded bolt) — taken jointly over every load case, since each case is evaluated separately on the same bolt group (shared centroid and polar J, its own direct and moment terms) and the governing case is simply whichever one peaks highest

Expressed with bolt coordinates as `torch` tensors, this whole formula is differentiable end to end — `autograd` produces the gradient of peak load with respect to every bolt's position, with no derivatives written by hand.

### Bearing / tear-out model (effective load)

A bolt's raw shear is only half the story: what actually fails first near an
edge is bearing / tear-out of the plate, and its capacity depends on the **clear
distance `lc` from the hole edge to the nearest material edge, measured in the
direction the bolt pushes on the plate** (AISC J3.10 in shape). "Nearest
material edge" means whichever comes first along that ray: the part's outer
profile, an existing hole or keepout, or an adjacent bolt hole.

**Sign convention — the bearing direction is `-load`.** The elastic method's
per-bolt resultant is the force the *plate* applies to the *bolt*, so it points
along the applied force `F`. Bearing and tear-out happen where the *bolt* pushes
back on the plate, which is the opposite direction. A tension plate pulled to
the left tears out at its right-hand end — the material between the hole and the
end edge on the side *opposite* the pull. So every clear distance is cast along
`-total_i,c`, i.e. `angle = atan2(-total_y, -total_x)`, and the adjacent-bolt
corridor uses that same reversed unit vector. The canvas tick is drawn in this
direction too and is labelled "bearing dir" so it is not read as the load.

Rather than introduce a second ceiling, the capacity is folded back into the
load, so the single load ceiling keeps meaning what it used to:

```
lc_i,c  = ray_distance(P_i, -dir(load_i,c)) - d/2     # clear distance where the bolt bears
k_i,c   = clamp(lc_i,c / (2 d), k_min, 1)             # capacity fraction, saturating at lc = 2d
L_eff   = |load_i,c| / k_i,c
peak    = max over (i, c) of L_eff
```

`d` is the bolt/hole diameter and `k_min` (0.05) keeps the barrier finite rather
than producing a division by zero. `L_eff` replaces the raw magnitude
everywhere the objective is taken — both the `logsumexp` soft peak and the hard
reported peak — while the raw peak stays in the result for reference.

Crucially `dir(load)` is the bolt's own resultant, which depends on the layout,
so autograd flows through the direction as well as the position; the model is
never detached.

Implementation: `app/geometry.py` precomputes, alongside the SDF, a **ray
distance field** `D[K, H, W]` (K = 32 directions, 256 cells on the long side)
against the *unshrunk* material boundary, sampled with one trilinear
`grid_sample` (bilinear in position, linear in angle, wrapping at 2π).
Adjacent-bolt blocking is computed live: bolt j shortens bolt i's tear-out path
to `dot(Pj - Pi, u) - d` whenever it sits inside a corridor of half-width `d`
around i's bearing ray, gated by a clamped ramp so the test stays
differentiable.

**The ray field is capped and min-pooled, because directional distance is
discontinuous.** Swing a ray a fraction of a degree, or shift its origin by a
fraction of a cell, and if it starts grazing *past* a small hole instead of
hitting it, the true answer jumps from a few millimetres to the width of the
plate. Interpolating the nodal values directly therefore does not merely blur
the field: it reports a long clear distance at points whose real clear distance
is short — an *unsafe* error, and the one that made every bolt read `k = 1`.
Two changes contain it:

* the field is clipped at `3d` when it is rasterized. `k` saturates at
  `lc = 2d`, i.e. a ray distance of `2.5d`, so nothing past that is visible to
  the model, and clipping bounds the dynamic range that interpolation can
  smear;
* the nodal field is reduced to a **per-cell minimum** over its 2×2×2 stencil
  (two in x, two in y, two in angle, wrapping), and it is that conservative cell
  field which is interpolated. The sampled value is then never longer than the
  shortest ray seen around the query, at the cost of being at most about one
  cell pessimistic.

On the sample plate this takes the field from a mean error of ~3.2 cells with
over-reports of up to +7 mm in the `lc < 2d` range, to a mean of ~0.45 cells
with **no over-reports at all**. The residual error is one-sided (the field
under-reports at grazing directions) and therefore always conservative.
`tests/test_optimizer.py` asserts both properties against an exact shapely ray
cast.

The hard edge clearance is unchanged and remains a floor; the bearing model does
the shaping above it, replacing "a fixed clearance decides quality" with a real
directional failure mode.

Note: a plain `max` has a sharp, non-smooth kink where the worst-loaded bolt switches. Gradients still exist almost everywhere, but a `logsumexp`-based soft-max can be substituted for a fully smooth objective if the hard max causes optimization to behave roughly near the kink.

## Constraints

Gradient descent has no built-in notion of a boundary it can't cross — every constraint has to be handled explicitly, either as:

- **Projection** — after each optimizer step, snap any bolt that landed outside the legal region back onto its nearest legal point
- **Penalty/barrier term** — add a term to the loss that grows sharply as a bolt approaches a forbidden region, so the gradient naturally repels it

**Hard constraints** (must never be violated — these are real failure modes, like bearing/tear-out near a boundary or a bolt landing inside a keepout zone):

- Keepout zones
- Minimum edge clearance
- Minimum bolt-to-bolt spacing

**Soft objective terms** (a preference the optimizer can trade off against peak load, not a hard floor):

- General tendency to favor bolts farther from the part edge, when not already covered by a hard clearance minimum

The distinction matters: a soft term can be traded away by the optimizer if it improves the primary objective enough; a hard constraint cannot.

## Implemented search: one batched run over all counts and seeds

The two loops below describe the problem structure, but the implementation
collapses them into a **single batched optimization**. All bolt counts in the
sweep and all random restarts live together in one `[B, N_max, 2]` position
tensor with a boolean slot mask saying which bolts are live for each layout,
so a batch of ~1000 diverse layouts spanning `n_min..n_max` is optimized by
one `Adam` instance in one `.backward()` per iteration. The seeds are
deliberately diverse (mostly farthest-point subsets of a uniform pool, plus
rings, grids and pure uniform noise), because restarts land in very different
local minima and the interesting question is usually *which* minima exist, not
just the single best one. The region constraint is a **rasterized signed
distance field** of the legal region sampled with a differentiable bilinear
`grid_sample`, so the inner loop never calls shapely; shapely is used once to
rasterize the field and once at the end for a hard projection snap. After
about a third of the iteration budget the worst half of the seeds *within each
bolt count* are culled (never across counts — peak load falls monotonically
with N, so a global cull would erase every low-N candidate). The run reports
the winning layout, a per-N summary of the best achievable peak at each count,
and the top-5 geometrically distinct layouts at the winning count, clustered by
a permutation-invariant signature.

## Optimization approach (inner loop — fixed bolt count)

For a fixed number of bolts N:

1. Initialize N bolt positions as a `torch` tensor (random or grid-seeded within the legal region)
2. Compute the combined loss = peak bolt load + weighted penalty terms for any constraint violations
3. Call `.backward()` to get the gradient with respect to bolt positions
4. Step with `torch.optim.Adam` (adaptive step size, generally more forgiving to tune than plain fixed-step gradient descent)
5. Repeat until the loss stops improving meaningfully
6. Check whether the resulting peak load clears the target ceiling

This inner loop is cheap — a handful of scalar coordinates, a few arithmetic operations per evaluation — and converges quickly on CPU. No GPU benefit for a single run at this scale.

## Outer loop — bolt count

Bolt count isn't a continuous quantity, so it can't be optimized by gradient descent directly. It stays as a simple outer sweep:

- Try N = 2, run the inner loop to convergence, check peak load against the ceiling
- If still over, try N = 3, and so on, until the ceiling is cleared
- Optionally run a few random restarts per N, since gradient descent can settle into a locally-good-but-not-globally-best layout

Because this sweep is many small, independent, cheap runs, it's naturally batchable — this is where a GPU could actually help (running many N values and/or many random starts in parallel), even though a single run doesn't benefit from one.

## Tooling / stack

- **`torch`** — autograd for the loss gradient, `torch.optim.Adam` for the inner-loop optimizer, `torch.logsumexp` if the hard-max objective needs smoothing
- **`shapely`** — polygon boolean and offset operations for deriving the legal placement region (boundary minus keepouts, inward edge offset)
- **`numpy`** — general array handling alongside torch tensors
- **`matplotlib`** — visualizing candidate layouts as the setup is tuned

Python over C# for this: the geometry + differentiable-optimization stack above is mature and interoperable in Python, and prototyping speed matters most while the objective/constraint shape is still being worked out.

## Open questions / future extensions

- What's the actual objective when trading off bolt count vs. margin: minimum count under the ceiling, maximum margin at a fixed count, or something else (e.g. minimizing total edge-distance-weighted material use)?
- How strongly should the edge-distance soft term be weighted relative to peak load — this likely needs some trial and error once real numbers are in play
- Penalty weight tuning in general (keepout, spacing) — cheap to iterate on with this setup, but not automatic
- Possible later upgrade: local-swap refinement on top of the optimized layout, to close any gap left by getting stuck in a locally-good arrangement
- Not covered by this spec at all: the separate pivot + shear pin joint, which isn't a bolt-group problem in the same sense (the pin carries the moment via lever arm from the pivot, not via a shared elastic-method load split)
