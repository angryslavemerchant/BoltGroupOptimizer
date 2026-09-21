"""Elastic (polar-moment) bolt-load model + batched torch/Adam layout search.

The search is a *single* batched run: several hundred to a few thousand diverse
seed layouts, spanning every bolt count in the requested sweep, are optimized
simultaneously by one Adam instance.  Layouts with different bolt counts share
one `[B, N_max, 2]` position tensor and are distinguished by a boolean slot
mask, so "the N sweep" and "the restarts" both collapse into the batch
dimension.  The region constraint is a rasterized signed distance field
(`app.geometry.RegionSDF`), so nothing in the inner loop calls shapely; shapely
is used once up front to rasterize and once at the end for a hard snap.
"""
import asyncio
import time
import math
import numpy as np
import shapely
import torch

from app.geometry import build_region_sdf, project_to_region, region_tolerance


# --------------------------------------------------------------------------
# Physics (batched)
# --------------------------------------------------------------------------

def batched_bolt_loads(P, M, force_points, force_vectors):
    """P: (B, N, 2) positions. M: (B, N) bool slot mask.

    force_points / force_vectors: (C, 2) tensors -- C *independent* load cases,
    each evaluated on its own against the same bolt group (no superposition).
    A bare (2,) tensor is accepted and read as a single case.

    Returns (total (B,C,N,2), mags (B,C,N)) with dead slots zeroed. Elastic/polar
    method exactly as in CLAUDE.md: direct shear F_c/n plus a moment term
    distributed in proportion to radius from the bolt-group centroid. Centroid
    and polar J depend only on the layout, so they are shared across cases.
    """
    FP = force_points.reshape(-1, 2)
    FV = force_vectors.reshape(-1, 2)
    B, N = M.shape
    C = FV.shape[0]

    Mf = M.to(P.dtype)
    n = Mf.sum(-1).clamp_min(1.0)                                   # (B,)
    centroid = (P * Mf.unsqueeze(-1)).sum(1) / n.unsqueeze(-1)      # (B,2)

    direct = FV.view(1, C, 2) / n.view(B, 1, 1)                     # (B,C,2)

    r = (P - centroid.unsqueeze(1)) * Mf.unsqueeze(-1)              # (B,N,2)
    J = (r * r).sum(-1).sum(-1).clamp_min(1e-9)                     # (B,) polar moment

    rf = FP.view(1, C, 2) - centroid.view(B, 1, 2)                  # (B,C,2)
    moment = rf[..., 0] * FV[:, 1].view(1, C) - rf[..., 1] * FV[:, 0].view(1, C)  # (B,C)

    perp = torch.stack([-r[..., 1], r[..., 0]], dim=-1)             # (B,N,2) = d_i * perp_unit_i
    moment_force = (moment / J.view(B, 1)).view(B, C, 1, 1) * perp.view(B, 1, N, 2)

    total = direct.unsqueeze(2) + moment_force                      # (B,C,N,2)
    total = total * Mf.view(B, 1, N, 1)
    # sqrt(.. + eps) rather than linalg.norm: norm has a NaN gradient at exactly
    # zero, which dead (all-zero) slots would otherwise hit every iteration.
    mags = torch.sqrt((total * total).sum(-1) + 1e-18) * Mf.view(B, 1, N)
    return total, mags


# --------------------------------------------------------------------------
# Bearing / tear-out (AISC J3.10 shaped, expressed as an *effective* load)
#
# A bolt's bearing capacity is proportional to the clear distance `lc` from the
# hole edge to the nearest material edge measured *in the direction the bolt
# pushes on the plate*, ramping linearly and saturating at lc = 2d.  Rather than
# introduce a second ceiling, that capacity fraction is folded back into the
# load: L_eff = |L| / k.  A bolt with full capacity (lc >= 2d) reports its raw
# load; one crowded against an edge reports a proportionally inflated one, and
# the single load ceiling still means what it used to.
#
# The direction is the bolt's own resultant, which depends on the layout, so
# autograd must flow through it -- it is deliberately not detached.
# --------------------------------------------------------------------------

LIMITER_OUTER, LIMITER_HOLE, LIMITER_BOLT = 0, 1, 2
LIMITER_NAMES = {LIMITER_OUTER: "outer edge", LIMITER_HOLE: "hole", LIMITER_BOLT: "adjacent bolt"}


def bearing_clear_distance(P, M, total, mags, fields, bolt_diameter, big=None):
    """Clear distance `lc` per (layout, case, bolt), plus the limiter that set it.

    total/mags: (B,C,N,2)/(B,C,N) bolt resultants from `batched_bolt_loads`.
    Returns (lc (B,C,N), limiter (B,C,N) long, angles (B,C,N)), where `angles` is
    the **bearing direction** -- the way the bolt pushes on the plate, i.e. the
    direction opposite the bolt's load resultant.

    Two competing limits are combined with a min:
      * the material edge -- the rasterized directional distance field, which
        already knows whether the crossing was the outer profile or a hole;
      * an adjacent bolt hole -- any live bolt j sitting in a corridor of
        half-width d around bolt i's load ray shortens the tear-out path to
        `dot(Pj - Pi, u) - d` (edge of hole to edge of hole).
    """
    B, C, N, _ = total.shape
    d = float(bolt_diameter)
    Mf = M.to(P.dtype)

    # atan2 has a NaN gradient at exactly (0,0), which dead slots would hit every
    # iteration, so nudge any ~zero resultant onto +x before taking the angle.
    tiny = (mags < 1e-9).to(P.dtype).unsqueeze(-1)
    safe = total + tiny * torch.tensor([1.0, 0.0], dtype=P.dtype, device=P.device)

    # SIGN: `total` is the force the *plate* applies to the bolt, so it points
    # along the applied load.  Bearing and tear-out happen where the *bolt* pushes
    # on the plate, which is the opposite direction -- a tension plate pulled to
    # the left tears out at the material between the hole and the right-hand end.
    # So every clear distance is measured along -total.
    bear = -safe                                                        # (B,C,N,2)
    u = bear / torch.sqrt((bear * bear).sum(-1, keepdim=True) + 1e-30)  # (B,C,N,2)
    angles = torch.atan2(bear[..., 1], bear[..., 0])                    # (B,C,N)

    Pc = P.unsqueeze(1).expand(B, C, N, 2)
    lc_mat, kind = fields.ray_distance(Pc, angles, with_kind=True)
    lc_mat = lc_mat - 0.5 * d

    if big is None:
        big = 4.0 * (abs(fields.maxx - fields.minx) + abs(fields.maxy - fields.miny)) + 1.0
    BIG = torch.tensor(float(big), dtype=P.dtype, device=P.device)

    rel = P.view(B, 1, 1, N, 2) - P.view(B, 1, N, 1, 2)                 # (B,1,N,N,2) = Pj - Pi
    u5 = u.view(B, C, N, 1, 2)
    along = (rel * u5).sum(-1)                                          # (B,C,N,N)
    perp = (rel[..., 0] * u5[..., 1] - rel[..., 1] * u5[..., 0]).abs()

    soft = max(0.1 * d, 1e-9)
    # Corridor gate: bolt j counts when it sits inside the half-width-d corridor
    # around bolt i's load ray AND is ahead of it. A clamped ramp rather than a
    # sigmoid, because a sigmoid never quite reaches 1 and the residual
    # `(1 - gate) * BIG` would bias every clear distance by a fraction of a unit;
    # this is exactly 1 well inside the corridor, exactly 0 outside it, and
    # linear (so differentiable) across the transition band.
    ramp = lambda z: torch.clamp(z / soft, min=0.0, max=1.0)
    gate = ramp(d - perp) * ramp(along - d)
    live_j = M.view(B, 1, 1, N).to(P.dtype)
    eye = torch.eye(N, dtype=torch.bool, device=P.device).view(1, 1, N, N)
    gate = gate * live_j * (~eye).to(P.dtype)

    cand = (along - d) + (1.0 - gate) * BIG
    lc_adj = cand.min(dim=-1).values                                    # (B,C,N)

    lc = torch.minimum(lc_mat, lc_adj)
    with torch.no_grad():
        limiter = torch.where(
            lc_adj < lc_mat,
            torch.full_like(kind, float(LIMITER_BOLT)),
            torch.where(kind > 0.5, torch.full_like(kind, float(LIMITER_HOLE)),
                        torch.zeros_like(kind)),
        ).long()
    return lc * Mf.view(B, 1, N), limiter, angles


def effective_loads(P, M, total, mags, bearing):
    """Raw magnitudes -> bearing-effective loads. `bearing` is None (model off) or
    {"fields", "d", "k_min"}. Returns (eff (B,C,N), lc, k, limiter) with lc/k/limiter
    None when the model is off."""
    if bearing is None:
        return mags, None, None, None
    d = float(bearing["d"])
    k_min = float(bearing.get("k_min", 0.05))
    lc, limiter, _ = bearing_clear_distance(P, M, total, mags, bearing["fields"], d)
    k = torch.clamp(lc / (2.0 * d), min=k_min, max=1.0)
    eff = mags / k
    return eff * M.to(P.dtype).view(M.shape[0], 1, M.shape[1]), lc, k, limiter


def _case_mask(mags, M):
    """Broadcast the (B,N) slot mask over the case dimension of a (B,C,N) tensor."""
    return M.view(M.shape[0], 1, M.shape[1]).expand_as(mags)


def peak_loads(mags, M, sharpness=200.0):
    """Smooth (logsumexp) and hard (max) peak per-bolt load over all live (case, bolt)
    entries -- the layout must survive every case, so the peak is taken jointly.

    `sharpness` is *relative*: the logsumexp temperature is `peak / sharpness`,
    with the peak taken per layout and detached.  A fixed absolute temperature
    would make the objective depend on the load units -- with loads of order
    1000 (N) an absolute `beta = 8` was effectively a hard max, while with loads
    of order 1 (kN, kip) the same beta biased the soft peak by tens of percent.
    Relative, the soft peak sits within `log(C*N) / sharpness` of the hard one
    (about 1.5 % at the default) whatever the units, and bolts within ~1 % of
    the peak share the gradient.
    """
    Mx = _case_mask(mags, M)
    flat = mags.reshape(mags.shape[0], -1)
    Mf = Mx.reshape(mags.shape[0], -1)
    neg = torch.where(Mf, torch.zeros_like(flat), torch.full_like(flat, -1e18))
    hard = torch.where(Mf, flat, torch.full_like(flat, -float("inf"))).max(dim=-1).values
    scale = hard.detach().abs()
    scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale)).clamp_min(1e-12)
    beta = (sharpness / scale).unsqueeze(-1)                                    # (B,1)
    soft = torch.logsumexp(flat * beta + neg, dim=-1) / beta.squeeze(-1)
    return soft, hard


def per_case_peaks(mags, M):
    """(B,C) peak load in each case, and (B,C) index of the bolt carrying it."""
    Mx = _case_mask(mags, M)
    masked = torch.where(Mx, mags, torch.full_like(mags, -float("inf")))
    vals, idx = masked.max(dim=-1)
    return vals, idx


def pairwise_distances(P):
    """(B,N,N) euclidean distances, safe to differentiate through (eps under sqrt)."""
    diff = P.unsqueeze(2) - P.unsqueeze(1)
    return torch.sqrt((diff * diff).sum(-1) + 1e-18)


def batched_loss(P, M, force_points, force_vectors, sdf, min_spacing,
                 edge_weight=0.0, edge_clamp=None,
                 sharpness=200.0, weights=None, bearing=None):
    """Full per-layout loss. Returns (loss_per_layout (B,), info dict of detached terms).

    With `bearing` supplied the objective is the peak *effective* load
    (raw load / bearing capacity fraction) rather than the raw peak, so the
    optimizer trades layout geometry against tear-out capacity directly.
    """
    Mf = M.to(P.dtype)
    total, mags = batched_bolt_loads(P, M, force_points, force_vectors)
    eff, lc, kfac, _ = effective_loads(P, M, total, mags, bearing)
    soft, hard = peak_loads(eff, M, sharpness)

    w = weights or {}
    w_region = w.get("region", 1.0)
    w_spacing = w.get("spacing", 1.0)

    # region: quadratic hinge on negative signed distance (i.e. depth outside)
    sd = sdf.sample(P)                                               # (B,N)
    region_pen = (torch.clamp(-sd, min=0.0) ** 2 * Mf).sum(-1)

    # spacing: quadratic hinge on shortfall, over live i<j pairs only
    B, N = M.shape
    d = pairwise_distances(P)
    pair_live = M.unsqueeze(2) & M.unsqueeze(1)
    triu = torch.triu(torch.ones(N, N, dtype=torch.bool, device=P.device), diagonal=1)
    pair_mask = (pair_live & triu).to(P.dtype)
    spacing_pen = ((torch.clamp(min_spacing - d, min=0.0) ** 2) * pair_mask).sum((-1, -2))

    loss = soft + w_region * region_pen + w_spacing * spacing_pen

    # soft terms
    if edge_weight:
        cl = edge_clamp if edge_clamp is not None else max(min_spacing, 1.0)
        loss = loss - edge_weight * (torch.clamp(sd, max=cl) * Mf).sum(-1)

    info = {
        "mags": mags.detach(),
        "eff": eff.detach(),
        "lc": None if lc is None else lc.detach(),
        "k": None if kfac is None else kfac.detach(),
        "hard": hard.detach(),
        "region_pen": region_pen.detach(),
        "spacing_pen": spacing_pen.detach(),
        "sdf": sd.detach(),
    }
    return loss, info


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

def sample_pool(region, count, rng, batch=4096, max_rounds=40):
    """Uniform points inside `region` by vectorized rejection sampling."""
    minx, miny, maxx, maxy = region.bounds
    out = []
    have = 0
    for _ in range(max_rounds):
        if have >= count:
            break
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        keep = shapely.contains_xy(region, xs, ys)
        if keep.any():
            out.append(np.stack([xs[keep], ys[keep]], axis=1))
            have += int(keep.sum())
    if not out:
        p = region.representative_point()
        return np.tile(np.array([[p.x, p.y]]), (count, 1))
    pool = np.concatenate(out, axis=0)
    if len(pool) < count:
        idx = rng.integers(0, len(pool), count)
        return pool[idx]
    return pool[:count]


def farthest_point_subset(pool, n, rng):
    """Greedy farthest-point sampling: maximally spread, so spacing-compliant early."""
    start = int(rng.integers(0, len(pool)))
    chosen = [start]
    d = np.linalg.norm(pool - pool[start], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(pool - pool[nxt], axis=1))
    return pool[chosen]


def _snap_all(pts, region):
    out = np.array(pts, dtype=float)
    inside = shapely.contains_xy(region, out[:, 0], out[:, 1])
    for i in np.flatnonzero(~inside):
        out[i, 0], out[i, 1] = project_to_region(out[i, 0], out[i, 1], region)
    return out


def ring_seed(region, center, radius, n, rot, rng):
    ang = rot + np.arange(n) * (2 * np.pi / n)
    pts = np.stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)], axis=1)
    return _snap_all(pts, region)


def grid_seed(region, pool, n, pitch, rot, rng):
    """n points from a rotated square lattice, clipped to the region."""
    minx, miny, maxx, maxy = region.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    k = int(max(2, math.ceil(max(maxx - minx, maxy - miny) / max(pitch, 1e-9)))) + 1
    ii, jj = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
    gx, gy = ii.ravel() * pitch, jj.ravel() * pitch
    c, s = math.cos(rot), math.sin(rot)
    rx, ry = cx + gx * c - gy * s, cy + gx * s + gy * c
    keep = shapely.contains_xy(region, rx, ry)
    cand = np.stack([rx[keep], ry[keep]], axis=1)
    if len(cand) < n:
        return farthest_point_subset(pool, n, rng)
    # compact cluster around a random anchor, so grids differ from uniform noise
    anchor = cand[int(rng.integers(0, len(cand)))]
    order = np.argsort(np.linalg.norm(cand - anchor, axis=1))
    return cand[order[:n]]


def _drop_near_fixed(pool, fixed, min_spacing):
    """Pool points at least `min_spacing` from every fixed bolt.

    Returns the filtered pool, or the original one if the filter emptied it (a
    region so tight that no legal free slot exists -- the spacing penalty then
    has to sort it out rather than the seeder silently producing nothing)."""
    if len(fixed) == 0 or len(pool) == 0:
        return pool
    d = np.linalg.norm(pool[:, None, :] - fixed[None, :, :], axis=2)
    keep = (d >= min_spacing).all(axis=1)
    return pool[keep] if keep.any() else pool


def build_seeds(region, n_min, n_max, seeds_per_n, min_spacing, rng, pool_size=600,
                fixed=None):
    """Returns (positions (B, n_max, 2) float64 array, mask (B, n_max) bool, ns (B,) int).

    `fixed` is an optional (F, 2) array of user-placed bolts. They occupy slots
    0..F-1 of every seed verbatim; only the remaining `n - F` slots are sampled,
    and they are drawn from a pool with everything within `min_spacing` of a
    fixed bolt removed, so no seed starts out violating spacing against a bolt
    the optimizer is not allowed to move.
    """
    fixed = (np.zeros((0, 2)) if fixed is None
             else np.asarray(fixed, dtype=float).reshape(-1, 2))
    F = len(fixed)
    pool = _drop_near_fixed(sample_pool(region, pool_size, rng), fixed, min_spacing)
    minx, miny, maxx, maxy = region.bounds
    span = max(maxx - minx, maxy - miny)
    rp = region.representative_point()
    center = (rp.x, rp.y)

    seeds, masks, ns = [], [], []
    seen = set()
    for n in range(n_min, n_max + 1):
        free = n - F
        made = 0
        attempts = 0
        while made < seeds_per_n and attempts < seeds_per_n * 4:
            attempts += 1
            if free <= 0:
                pts = fixed[:n].copy()
                if made:  # nothing left to vary -- one seed is all there is
                    break
            else:
                u = rng.random()
                if u < 0.60:
                    sub = pool[rng.integers(0, len(pool), min(len(pool), 200))]
                    pts = farthest_point_subset(sub, free, rng)
                elif u < 0.80:
                    if rng.random() < 0.5:
                        radius = span * float(rng.choice([0.15, 0.25, 0.35, 0.45]))
                        pts = ring_seed(region, center, radius, free, float(rng.random() * 2 * np.pi), rng)
                    else:
                        pitch = max(min_spacing, span * float(rng.choice([0.12, 0.2, 0.3])))
                        pts = grid_seed(region, pool, free, pitch, float(rng.random() * np.pi / 2), rng)
                else:
                    pts = pool[rng.integers(0, len(pool), free)]
                if F:
                    # ring/grid seeds are built from geometry, not from the
                    # filtered pool, so re-check them and swap any offender for
                    # a pool point that is known to clear every fixed bolt.
                    pts = np.asarray(pts, dtype=float).copy()
                    bad = (np.linalg.norm(pts[:, None, :] - fixed[None, :, :], axis=2)
                           < min_spacing).any(axis=1)
                    for j in np.flatnonzero(bad):
                        pts[j] = pool[int(rng.integers(0, len(pool)))]
                    pts = np.concatenate([fixed, pts], axis=0)

            sig = tuple(np.round(np.sort(np.linalg.norm(pts - pts.mean(0), axis=1)) /
                                 max(span * 1e-3, 1e-9)).astype(int))
            if sig in seen:
                continue
            seen.add(sig)

            slot = np.zeros((n_max, 2))
            slot[:n] = pts
            # park dead slots on the region centroid so they never sit at a
            # degenerate (0,0) that could blow up distances
            slot[n:] = np.array(center)
            m = np.zeros(n_max, dtype=bool)
            m[:n] = True
            seeds.append(slot)
            masks.append(m)
            ns.append(n)
            made += 1

    return np.array(seeds), np.array(masks), np.array(ns)


# --------------------------------------------------------------------------
# Post-processing helpers
# --------------------------------------------------------------------------

def spacing_flags(P, M, min_spacing, tol=1e-6):
    """Minimum-spacing compliance: (B,N) per bolt and (B,) per layout.

    A bolt is flagged when *any* other live bolt is closer than `min_spacing`, so
    both members of an offending pair read as violations -- which is what the UI
    wants to colour, whereas the layout-level flag is the plain conjunction.
    """
    d = pairwise_distances(P)
    N = M.shape[1]
    eye = torch.eye(N, dtype=torch.bool, device=P.device).view(1, N, N)
    live_pair = (M.unsqueeze(2) & M.unsqueeze(1)) & ~eye
    viol = live_pair & (d < float(min_spacing) - tol)
    return (~viol.any(-1)) & M, ~viol.any(-1).any(-1)


def score_layouts(positions, masks, force_points, force_vectors, case_names,
                  bearing=None, min_spacing=0.0, fixed_count=0,
                  region_ok=None, per_bolt_region_ok=None,
                  dtype=torch.float64, device=None):
    """Score a batch of layouts -> one result payload dict per layout.

    This is the *single* scoring path. `run_optimization` calls it on its final
    (post-snap) batch, and `POST /evaluate` calls it with a batch of one, so a
    hand-dragged layout is measured by exactly the same code -- and reports
    exactly the same numbers -- as an optimized one.

    positions: (B, N, 2), masks: (B, N) bool. `bearing` is the dict
    `effective_loads` takes (or None). `fixed_count` marks slots 0..F-1 as
    user-pinned. `region_ok` (B,) / `per_bolt_region_ok` (B, N) are the shapely
    region checks, done by the caller (this function never touches shapely);
    when omitted, feasibility considers spacing only.

    Each payload carries n, positions, fixed_flags, peak_load (effective),
    raw_peak_load, per_bolt_loads, per_bolt rows, per_case peaks, governing_case
    and feasible -- the shape `/result`, the alternatives list and the results
    table all already consume.
    """
    P = torch.as_tensor(np.asarray(positions, dtype=float), dtype=dtype, device=device)
    M = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=device)
    FP = torch.as_tensor(force_points, dtype=dtype, device=device).reshape(-1, 2)
    FV = torch.as_tensor(force_vectors, dtype=dtype, device=device).reshape(-1, 2)
    F = int(fixed_count)

    with torch.no_grad():
        total, mags = batched_bolt_loads(P, M, FP, FV)
        eff, lc_t, k_t, lim_t = effective_loads(P, M, total, mags, bearing)
        # Per (case, bolt) bearing *direction* -- the way the bolt pushes on the
        # plate, i.e. -total normalized -- needed for the canvas to draw one tick
        # per force case per bolt (not just the governing case's tick). Same sign
        # convention as `bearing_clear_distance`; computed unconditionally since
        # it costs nothing extra and does not depend on the bearing model being on.
        tiny = (mags < 1e-9).to(P.dtype).unsqueeze(-1)
        safe_total = total + tiny * torch.tensor([1.0, 0.0], dtype=P.dtype, device=P.device)
        bear = -safe_total
        dirs_t = bear / torch.sqrt((bear * bear).sum(-1, keepdim=True) + 1e-30)  # (B,C,N,2)
        _, hard = peak_loads(eff, M)
        _, raw_hard = peak_loads(mags, M)
        case_peak, case_bolt = per_case_peaks(eff, M)
        case_raw_peak, _ = per_case_peaks(mags, M)
        hard_np = hard.cpu().numpy()
        raw_hard_np = raw_hard.cpu().numpy()
        # The per-bolt row reported to the user is that bolt's worst case, chosen
        # by *effective* load -- the quantity the ceiling is now compared against.
        wc = eff.argmax(dim=1).unsqueeze(1)                             # (B,1,N)
        mags_np = mags.gather(1, wc).squeeze(1).cpu().numpy()
        eff_np = eff.gather(1, wc).squeeze(1).cpu().numpy()
        lc_np = None if lc_t is None else lc_t.gather(1, wc).squeeze(1).cpu().numpy()
        k_np = None if k_t is None else k_t.gather(1, wc).squeeze(1).cpu().numpy()
        lim_np = None if lim_t is None else lim_t.gather(1, wc).squeeze(1).cpu().numpy()
        case_peak_np = case_peak.cpu().numpy()
        case_raw_np = case_raw_peak.cpu().numpy()
        case_bolt_np = case_bolt.cpu().numpy()
        sp_bolt_t, sp_layout_t = spacing_flags(P, M, min_spacing)
        sp_bolt = sp_bolt_t.cpu().numpy()
        spacing_ok = sp_layout_t.cpu().numpy()
        dirs_np = dirs_t.cpu().numpy()

    Mn = M.cpu().numpy()
    Pn = P.cpu().numpy()
    layouts = []
    for b in range(Pn.shape[0]):
        live = np.flatnonzero(Mn[b])
        slot_to_live = {int(s): j for j, s in enumerate(live)}
        per_case = [{
            "name": case_names[c],
            # `peak` is the effective (bearing-adjusted) peak -- the one compared
            # against the ceiling; the raw elastic peak is kept alongside it.
            "peak": float(case_peak_np[b, c]),
            "raw_peak": float(case_raw_np[b, c]),
            "governing_bolt_index": slot_to_live.get(int(case_bolt_np[b, c]), 0),
            # unit bearing direction for every live bolt, in per_bolt/positions
            # order -- lets the canvas draw one tick per case per bolt, not just
            # the governing case's tick.
            "bolt_dirs": dirs_np[b, c, live].tolist(),
        } for c in range(len(case_names))]
        governing = max(per_case, key=lambda p: p["peak"])["name"] if per_case else None
        per_bolt = []
        for j, s in enumerate(live):
            row = {"index": j, "load": float(mags_np[b, s]),
                   "effective_load": float(eff_np[b, s]),
                   "fixed": bool(s < F),
                   "spacing_ok": bool(sp_bolt[b, s])}
            if lc_np is not None:
                row["lc"] = float(lc_np[b, s])
                row["k"] = float(k_np[b, s])
                row["limiter"] = LIMITER_NAMES.get(int(lim_np[b, s]), "")
            if per_bolt_region_ok is not None:
                row["in_region"] = bool(per_bolt_region_ok[b][s])
            per_bolt.append(row)
        reg_ok = True if region_ok is None else bool(region_ok[b])
        layouts.append({
            "n": int(Mn[b].sum()),
            "positions": Pn[b, live].tolist(),
            "fixed_flags": [bool(s < F) for s in live],
            "peak_load": float(hard_np[b]),
            "raw_peak_load": float(raw_hard_np[b]),
            "per_bolt_loads": mags_np[b, live].tolist(),
            "per_bolt": per_bolt,
            "bearing": bool(bearing is not None),
            "per_case": per_case,
            "governing_case": governing,
            "spacing_ok": bool(spacing_ok[b]),
            "region_ok": reg_ok,
            "feasible": bool(spacing_ok[b] and reg_ok),
        })
    return layouts


def _signature(pts):
    """Permutation-invariant layout signature: sorted distances to the centroid."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0)
    c = pts.mean(axis=0)
    return np.sort(np.linalg.norm(pts - c, axis=1))


def pick_distinct(cands, threshold, k=5):
    """Greedy: walk candidates best-first, keep one representative per cluster."""
    out = []
    sigs = []
    for c in cands:
        s = _signature(np.asarray(c["positions"]))
        if all(np.linalg.norm(s - t) > threshold for t in sigs):
            out.append(c)
            sigs.append(s)
            if len(out) >= k:
                break
    return out


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------

def normalize_forces(forces):
    """Accept a list of force-case dicts (or one (point, vector) pair) -> list of dicts.

    Each case is {name, x, y, fx, fy}: its own point of application and its own
    vector, evaluated independently of the others.
    """
    out = []
    for i, f in enumerate(forces):
        if isinstance(f, dict):
            out.append({
                "name": f.get("name") or f"Case {i + 1}",
                "x": float(f["x"]), "y": float(f["y"]),
                "fx": float(f["fx"]), "fy": float(f["fy"]),
            })
        else:  # ((x, y), (fx, fy))
            (x, y), (fx, fy) = f
            out.append({"name": f"Case {i + 1}", "x": float(x), "y": float(y),
                        "fx": float(fx), "fy": float(fy)})
    return out


def normalize_fixed_bolts(fixed_bolts):
    """Accept [{x, y}] or [(x, y)] -> (F, 2) float array."""
    if not fixed_bolts:
        return np.zeros((0, 2))
    out = []
    for b in fixed_bolts:
        if isinstance(b, dict):
            out.append((float(b["x"]), float(b["y"])))
        else:
            out.append((float(b[0]), float(b[1])))
    return np.array(out, dtype=float).reshape(-1, 2)


async def run_optimization(region, forces, settings,
                           stream_every=10, should_stop=None, material_region=None,
                           fixed_bolts=None, fields=None):
    """Async generator yielding progress dicts and one final dict with status='done'.

    `forces` is a list of independent load cases, each {name, x, y, fx, fy}. Every
    case is evaluated separately on the same bolt group (never superposed) and the
    objective is the peak per-bolt load over all (case, bolt) pairs.

    settings keys: min_spacing, load_ceiling, n_min, n_max, seeds_per_n, iterations,
      edge_weight (optional), edge_clearance (optional,
      for the edge-preference clamp), sdf_resolution (optional), use_gpu (optional),
      seed (optional), bolt_diameter / bearing_enabled / k_min / ray_dirs /
      ray_resolution (optional, the bearing / tear-out model).

    `material_region` is the placement region *before* the edge-clearance shrink,
    i.e. the real material edge. The bearing model measures its clear distances
    against it; without it the bearing model cannot run and is skipped.

    `fixed_bolts` are user-placed bolts that never move. They are full members of
    the group -- they carry load, count toward the centroid and polar moment, and
    participate in spacing and bearing -- so they occupy slots 0..F-1 of every
    layout in the batch, live and pinned. `n` still counts *total* bolts, so the
    sweep is clamped to n >= F.

    `fields` is an optional prebuilt `RegionFields` for exactly this region /
    material / settings (the server caches one for the drag-evaluate loop); when
    it matches the bearing configuration it is reused instead of rasterizing
    the fields a second time.
    """
    cases = normalize_forces(forces)
    if not cases:
        yield {"status": "error", "message": "No force cases defined."}
        return

    if region.is_empty:
        yield {"status": "error",
               "message": "Legal placement region is empty. Loosen clearance/spacing or remove keepouts."}
        return

    min_spacing = float(settings["min_spacing"])
    ceiling = float(settings["load_ceiling"])
    n_min = int(settings["n_min"])
    n_max = int(settings["n_max"])

    fixed = normalize_fixed_bolts(fixed_bolts)
    F = len(fixed)
    n_min_req, n_max_req = n_min, n_max
    # A layout needs at least one bolt: an empty layout has a peak of -inf,
    # which would win every comparison and then fail to serialize.
    n_min = max(n_min, F, 1)
    n_max = max(n_max, n_min)
    clamp_note = ""
    if (n_min, n_max) != (n_min_req, n_max_req):
        why = f"{F} fixed bolt(s) placed" if F and n_min == F else "N must be at least 1"
        clamp_note = (f" {why}, so the sweep was clamped to "
                      f"N={n_min}..{n_max} (requested {n_min_req}..{n_max_req}).")

    seeds_per_n = int(settings.get("seeds_per_n", settings.get("restarts", 80)) or 1)
    iterations = int(settings["iterations"])
    edge_weight = float(settings.get("edge_weight", 0.0) or 0.0)
    edge_clearance = float(settings.get("edge_clearance", 0.0) or 0.0)
    sdf_resolution = int(settings.get("sdf_resolution", 384) or 384)
    seed = int(settings.get("seed", 0) or 0)

    # Bearing / tear-out. `bearing_enabled` defaults on, but the model is a no-op
    # without a bolt diameter to measure 2d saturation against and without the
    # unshrunk material outline to measure clear distance to -- callers that
    # supply neither (e.g. the pure-physics tests) keep the raw-load objective.
    bolt_diameter = float(settings.get("bolt_diameter", 0.0) or 0.0)
    k_min = float(settings.get("k_min", 0.05) or 0.05)
    ray_dirs = int(settings.get("ray_dirs", 32) or 32)
    ray_resolution = int(settings.get("ray_resolution", 256) or 256)
    bearing_on = (bool(settings.get("bearing_enabled", True))
                  and bolt_diameter > 0.0
                  and material_region is not None
                  and not material_region.is_empty)

    cuda_available = torch.cuda.is_available()
    use_gpu = bool(settings.get("use_gpu", False)) and cuda_available
    device = torch.device("cuda" if use_gpu else "cpu")
    dtype = torch.float64

    def _stop():
        return bool(should_stop and should_stop())

    rng = np.random.default_rng(seed)

    t_fields = time.perf_counter()
    if fields is not None and bool(fields.has_ray_field) == bool(bearing_on):
        sdf = fields
        if sdf.device != device or sdf.dtype != dtype:
            sdf = sdf.to(device=device, dtype=dtype)
    else:
        sdf = build_region_sdf(region, resolution=sdf_resolution, device=device, dtype=dtype,
                               material_region=material_region,
                               n_dirs=ray_dirs if bearing_on else 0,
                               ray_resolution=ray_resolution,
                               # k saturates at lc = 2d, i.e. a ray distance of 2.5d;
                               # a little headroom past that is all the model can use.
                               ray_cap=(3.0 * bolt_diameter) if bearing_on else None)
    field_build_s = time.perf_counter() - t_fields
    bearing = ({"fields": sdf, "d": bolt_diameter, "k_min": k_min}
               if (bearing_on and sdf.has_ray_field) else None)

    seeds, masks, ns = build_seeds(region, n_min, n_max, seeds_per_n, min_spacing, rng,
                                   fixed=fixed)
    if len(seeds) == 0:
        yield {"status": "error", "message": "Could not generate any seed layouts in the region."}
        return

    P = torch.tensor(seeds, dtype=dtype, device=device, requires_grad=True)
    M = torch.tensor(masks, dtype=torch.bool, device=device)
    N_vec = torch.tensor(ns, dtype=torch.long, device=device)

    force_points = torch.tensor([[c["x"], c["y"]] for c in cases], dtype=dtype, device=device)
    force_vectors = torch.tensor([[c["fx"], c["fy"]] for c in cases], dtype=dtype, device=device)
    case_names = [c["name"] for c in cases]
    fixed_t = torch.tensor(fixed, dtype=dtype, device=device) if F else None

    force_mag = max((math.hypot(c["fx"], c["fy"]) for c in cases), default=0.0) or 1.0
    # Penalty weights are scaled so a full-`min_spacing` violation costs a
    # multiple of the applied force -- keeps them meaningful whatever the
    # drawing units and load magnitudes happen to be.
    pen_scale = force_mag / max(min_spacing, 1e-6) ** 2
    weights = {"region": 20.0 * pen_scale, "spacing": 5.0 * pen_scale}
    edge_clamp = max(edge_clearance * 2.0, min_spacing, 1e-9)

    # Adam's step is in drawing units, so it has to be sized from the drawing,
    # never from an absolute constant: a floor of 1.0 was a sensible 1 mm on a
    # metric plate and a wild 1 inch on an imperial one.
    rminx, rminy, rmaxx, rmaxy = region.bounds
    region_span = max(rmaxx - rminx, rmaxy - rminy, 1e-9)
    lr = max(min_spacing * 0.1, region_span * 1e-3)

    optim = torch.optim.Adam([P], lr=lr)

    cull_at = max(1, iterations // 3)
    stopped = False

    def _loss():
        return batched_loss(P, M, force_points, force_vectors, sdf, min_spacing,
                            edge_weight=edge_weight, edge_clamp=edge_clamp,
                            weights=weights, bearing=bearing)

    def _pin():
        """Hold the fixed bolts exactly where the user put them.

        Belt and braces: the gradient is zeroed *and* the coordinates are
        rewritten after the step. Zeroing from iteration 0 does keep Adam's
        moment buffers at zero for these slots today, but that is a property of
        never having seen a gradient -- anything that ever gives them one (a
        re-seed, a warm start, a future penalty applied before the mask) leaves
        stale momentum that keeps nudging a zero-grad parameter for many steps.
        The rewrite makes the guarantee unconditional; the zeroing keeps the
        rewrite from having to fight the optimizer every iteration.
        """
        if fixed_t is not None:
            with torch.no_grad():
                P[:, :F] = fixed_t

    it = 0
    for it in range(iterations):
        if _stop():
            stopped = True
            break

        optim.zero_grad()
        loss, info = _loss()
        loss.sum().backward()
        if fixed_t is not None and P.grad is not None:
            P.grad[:, :F] = 0.0
        optim.step()
        _pin()

        if it == cull_at and P.shape[0] > 8:
            with torch.no_grad():
                _, info2 = _loss()
                score = info2["hard"] + weights["region"] * info2["region_pen"] \
                    + weights["spacing"] * info2["spacing_pen"]
                # Cull *within each bolt count*, never across them: peak load
                # falls monotonically with n, so a global cull would wipe out
                # every low-n layout and both lose the per-n comparison the
                # user wants and bias the winner toward needlessly many bolts.
                keep_parts = []
                for nv in torch.unique(N_vec):
                    idx = torch.nonzero(N_vec == nv, as_tuple=False).flatten()
                    order = idx[torch.argsort(score[idx])]
                    keep_parts.append(order[: max(4, len(order) // 2)])
                keep = torch.cat(keep_parts)
            newP = P.detach()[keep].clone().requires_grad_(True)
            M = M[keep]
            N_vec = N_vec[keep]
            P = newP
            # fresh Adam: its per-parameter moment buffers are tied to the old
            # (larger) tensor and cannot be re-indexed meaningfully after a cull
            optim = torch.optim.Adam([P], lr=lr)
            # `info` above refers to the pre-cull batch; refresh it so a progress
            # frame on this same iteration indexes the surviving layouts.
            with torch.no_grad():
                _, info = _loss()

        if it % stream_every == 0 or it == iterations - 1:
            with torch.no_grad():
                hard = info["hard"]
                feas = info["region_pen"] + info["spacing_pen"]
                rank = hard + weights["region"] * info["region_pen"] + weights["spacing"] * info["spacing_pen"]
                per_n_best = []
                bi = None
                for n in range(n_min, n_max + 1):
                    sel = torch.nonzero(N_vec == n, as_tuple=False).flatten()
                    if len(sel) == 0:
                        continue
                    local = sel[torch.argmin(rank[sel])]
                    peak_n = float(hard[sel].min().item())
                    per_n_best.append({"n": n, "peak": peak_n})
                    # same rule as the final selection: smallest n under the ceiling
                    if bi is None and float(hard[local].item()) <= ceiling:
                        bi = int(local.item())
                if bi is None:
                    bi = int(torch.argmin(rank).item())
                bn = int(N_vec[bi].item())
                pos = P.detach()[bi][M[bi]].cpu().numpy().tolist()
            yield {
                "status": "progress",
                "iteration": it,
                "iterations": iterations,
                "batch_size": int(P.shape[0]),
                "best_peak": float(hard[bi].item()),
                "best_n": bn,
                "best_positions": pos,
                "per_n_best": per_n_best,
            }
            await asyncio.sleep(0)

    # ---- hard snap + feasibility, outside the differentiable loop ----
    with torch.no_grad():
        Pn = P.detach().cpu().numpy().copy()
        Mn = M.cpu().numpy()
        Nn = N_vec.cpu().numpy()

    for b in range(Pn.shape[0]):
        # fixed bolts (slots 0..F-1) are excluded from the snap: they were
        # validated against the region when the user placed them, and moving
        # them here -- even by a rounding-sized nudge -- would break the promise
        # that the reported coordinates are exactly the ones the user chose.
        live = np.flatnonzero(Mn[b])
        live = live[live >= F]
        if not len(live):
            continue
        pts = Pn[b, live]
        inside = shapely.contains_xy(region, pts[:, 0], pts[:, 1])
        for j in np.flatnonzero(~inside):
            x, y = project_to_region(pts[j, 0], pts[j, 1], region)
            Pn[b, live[j]] = (x, y)

    # Region feasibility is checked against the real polygon, not the raster --
    # a snapped-to-boundary bolt reads as slightly negative under bilinear
    # interpolation of the SDF even though it is legally on the boundary.
    tol = region_tolerance(region)
    region_check = region.buffer(tol)
    flat = Pn.reshape(-1, 2)
    ok_flat = shapely.contains_xy(region_check, flat[:, 0], flat[:, 1]).reshape(Pn.shape[:2])
    region_ok = np.all(ok_flat | ~Mn, axis=1)

    layouts = score_layouts(
        Pn, Mn, force_points, force_vectors, case_names,
        bearing=bearing, min_spacing=min_spacing, fixed_count=F,
        region_ok=region_ok, per_bolt_region_ok=ok_flat,
        dtype=dtype, device=device,
    )

    per_n = []
    for n in range(n_min, n_max + 1):
        group = [l for l in layouts if l["n"] == n]
        feas = [l for l in group if l["feasible"]]
        per_n.append({
            "n": n,
            "seeds": len(group),
            "feasible": len(feas),
            "best_peak": min((l["peak_load"] for l in feas), default=None),
            "meets_ceiling": bool(feas and min(l["peak_load"] for l in feas) <= ceiling),
        })

    feasible = [l for l in layouts if l["feasible"]]
    pool_for_pick = feasible if feasible else layouts
    winner = None
    ceiling_met = False
    for n in range(n_min, n_max + 1):
        cands = [l for l in pool_for_pick if l["n"] == n and l["peak_load"] <= ceiling]
        if cands:
            winner = min(cands, key=lambda l: l["peak_load"])
            ceiling_met = True
            break
    if winner is None and pool_for_pick:
        winner = min(pool_for_pick, key=lambda l: l["peak_load"])

    if winner is None:
        yield {"status": "done", "success": False, "stopped": stopped, "result": None,
               "per_n": per_n, "alternatives": [], "ceiling_met": False,
               "device": str(device), "cuda_available": cuda_available,
               "message": "No layout survived the search."}
        return

    same_n = sorted([l for l in pool_for_pick if l["n"] == winner["n"]],
                    key=lambda l: l["peak_load"])
    alternatives = pick_distinct(same_n, threshold=max(min_spacing * 0.5, 1e-9), k=5)

    result = {k: winner[k] for k in ("n", "positions", "peak_load", "raw_peak_load",
                                     "per_bolt_loads", "per_bolt", "bearing",
                                     "per_case", "governing_case", "fixed_flags")}
    result["ceiling_met"] = ceiling_met

    if stopped:
        message = f"Stopped early; showing best layout found so far (N={result['n']})."
    elif ceiling_met:
        message = (f"N={result['n']} clears the ceiling "
                   f"({result['peak_load']:.2f} ≤ {ceiling:.2f}); "
                   f"searched {len(layouts)} layouts in one batched run.")
    else:
        message = (f"Ceiling not met up to n_max={n_max}; "
                   f"returning best found (N={result['n']}).")
    message += clamp_note

    yield {
        "status": "done",
        "success": ceiling_met,
        "stopped": stopped,
        "result": result,
        "ceiling_met": ceiling_met,
        "alternatives": alternatives,
        "cases": case_names,
        "per_n": per_n,
        "batch_size": len(layouts),
        "iterations_run": it + 1,
        "bearing": bool(bearing is not None),
        "n_fixed": F,
        "n_min": n_min,
        "n_max": n_max,
        "n_clamped": bool(clamp_note),
        "bolt_diameter": bolt_diameter if bearing is not None else None,
        "field_build_s": field_build_s,
        "device": str(device),
        "cuda_available": cuda_available,
        "message": message,
    }
