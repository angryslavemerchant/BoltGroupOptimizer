"""Legal placement region: part polygon minus keepouts, shrunk by edge clearance."""
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union, nearest_points


def build_part_polygon(part_loop, hole_loops):
    """part_loop: list of (x,y). hole_loops: loops fully inside part_loop, become holes."""
    shell = Polygon(part_loop)
    if not shell.is_valid:
        shell = shell.buffer(0)
    holes = []
    for h in hole_loops:
        hp = Polygon(h)
        if not hp.is_valid:
            hp = hp.buffer(0)
        if shell.contains(hp.representative_point()):
            holes.append(list(h))
    if holes:
        shell = Polygon(part_loop, holes)
    return shell


def build_material_region(part_loop, keepout_loops, other_loops=None):
    """The placement region *before* the edge-clearance shrink.

    This is the real material edge the bearing / tear-out model measures clear
    distance to: the part outline, minus every loop inside the part that is a
    real hole (removed material).  A loop marked **keepout** is solid material
    -- it is a placement restriction, not a hole -- so it is deliberately *not*
    subtracted here; `keepout_loops` is accepted for call-site symmetry with
    `build_legal_region` but is unused.  A loop marked **ignore** never reaches
    this function at all (the caller drops it before building `other_loops`).
    `build_legal_region` is this region minus the keepouts, shrunk inward by the
    edge clearance.
    """
    part_poly = Polygon(part_loop)
    if not part_poly.is_valid:
        part_poly = part_poly.buffer(0)

    subtract_polys = []

    if other_loops:
        for loop in other_loops:
            lp = Polygon(loop)
            if not lp.is_valid:
                lp = lp.buffer(0)
            if lp.is_empty:
                continue
            if part_poly.contains(lp.representative_point()):
                subtract_polys.append(lp)

    region = part_poly
    if subtract_polys:
        union_subtract = unary_union(subtract_polys)
        region = region.difference(union_subtract)

    return region


def build_legal_region(part_loop, keepout_loops, edge_clearance, other_loops=None):
    """
    part_loop: outer boundary points.
    keepout_loops: loops the user flagged as keepout zones. Unlike a hole, a
                   keepout is *solid material* -- it is not subtracted from
                   `build_material_region` (so it is not a bearing edge and does
                   not affect the ray field), but it is subtracted here, from the
                   legal *placement* region, before the edge-clearance shrink --
                   so a keepout gets exactly the same inward buffer as the outer
                   edge, and bolts don't sit exactly on the keepout line.
    other_loops: every other closed loop from the DXF that is a real hole (not
                 the part, not a keepout, not marked ignore). Any of these fully
                 inside the part polygon represents real material removed and is
                 treated as a hole, per spec.
    Returns a shapely geometry (Polygon or MultiPolygon), possibly empty.
    """
    region = build_material_region(part_loop, [], other_loops=other_loops)

    if keepout_loops:
        keepout_polys = []
        for k in keepout_loops:
            kp = Polygon(k)
            if not kp.is_valid:
                kp = kp.buffer(0)
            keepout_polys.append(kp)
        region = region.difference(unary_union(keepout_polys))

    if edge_clearance and edge_clearance > 0:
        region = region.buffer(-edge_clearance)

    return region


def region_to_geojson_like(region):
    """Return a list of polygons, each {exterior: [...], holes: [[...], ...]} for JSON transport."""
    polys = []
    if region.is_empty:
        return polys
    geoms = list(region.geoms) if isinstance(region, MultiPolygon) else [region]
    for g in geoms:
        if g.is_empty:
            continue
        exterior = list(g.exterior.coords)
        holes = [list(interior.coords) for interior in g.interiors]
        polys.append({"exterior": exterior, "holes": holes})
    return polys


def region_tolerance(region):
    """A hair of slack for containment tests, scaled to the region's size.

    A bolt snapped onto the region boundary (or onto a hole centre that lands
    exactly on it) must not be rejected by floating-point noise, so every
    "is this bolt legal" check buffers the region by this much first.
    """
    if region.is_empty:
        return 1e-6
    minx, miny, maxx, maxy = region.bounds
    return max(1e-6, max(maxx - minx, maxy - miny) * 1e-6)


def project_to_region(x, y, region):
    """Nearest point in region to (x, y). If already inside, returns (x, y) unchanged."""
    pt = Point(x, y)
    if region.is_empty:
        return x, y
    if region.contains(pt):
        return x, y
    nearest = nearest_points(region, pt)[0]
    return nearest.x, nearest.y


# ---------------------------------------------------------------------------
# Rasterized signed distance field for the legal region.
#
# The inner optimization loop must not touch shapely (per-point python calls
# dominate the runtime and are not differentiable). Instead the region is
# rasterized once into a signed distance field -- positive inside, negative
# outside, in drawing units -- which is sampled with a differentiable bilinear
# `grid_sample`, so the region constraint becomes an ordinary penalty term with
# a usable gradient everywhere.
# ---------------------------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn.functional as F
import shapely


class RegionFields:
    """Rasterized fields over one shared grid + affine, sampled differentiably.

    Two fields live here:

    * ``grid`` -- the signed distance field of the *legal* region (positive
      inside, negative outside), used as the hard placement constraint.
    * ``ray``  -- an optional directional distance field ``[K, ny, nx]``: for
      each of K evenly spaced directions, the distance from a cell centre along
      that direction to the first crossing of the *unshrunk material* boundary.
      This is what the bearing / tear-out model measures ``lc`` against, so it
      deliberately uses the real material edge rather than the clearance-shrunk
      region.  ``ray_kind`` records which kind of boundary produced each entry
      (0 = outer edge, 1 = interior hole / keepout) so the governing limiter can
      be reported.

    ``grid[j, i]`` is the value at world point (x_i, y_j), with
    x = linspace(minx, maxx, nx) and y = linspace(miny, maxy, ny), i.e. the grid
    samples sit exactly on the bbox corners -- which is what
    ``align_corners=True`` in ``grid_sample`` assumes.  The ray field shares the
    bbox (hence the affine mapping) but may be rasterized at a coarser
    resolution, since it costs K analytic ray casts per cell.

    **Why the ray field is min-pooled before it is sampled.**  Unlike the SDF,
    directional distance is *discontinuous*: swing a ray a fraction of a degree,
    or shift its origin by a fraction of a cell, and if it starts grazing past a
    small hole instead of hitting it the answer jumps from a few millimetres to
    the width of the plate.  Interpolating the nodal values straight away
    therefore does not merely blur the field, it reports a long clear distance at
    points whose true clear distance is short -- an *unsafe* error in a tear-out
    model, and the one that made every bolt read k = 1.  So the nodal field is
    reduced to a per-cell minimum over its 2x2x2 stencil (two in x, two in y, two
    in angle, wrapping) and it is that conservative cell field which gets
    interpolated.  The sampled value is then never longer than the shortest ray
    seen anywhere around the query, at the price of being at most one cell / one
    direction step pessimistic.
    """

    def __init__(self, grid, minx, miny, maxx, maxy, ray=None, ray_kind=None):
        self.grid = grid  # (ny, nx) tensor
        self.minx, self.miny, self.maxx, self.maxy = float(minx), float(miny), float(maxx), float(maxy)
        self._g = grid.unsqueeze(0).unsqueeze(0)  # (1,1,ny,nx) for grid_sample
        self.ray = ray                            # (K, ny2, nx2) or None
        self.ray_kind = ray_kind                  # (K, ny2, nx2) float 0/1 or None
        # The direction axis becomes the depth axis of a volume, so one
        # *trilinear* grid_sample interpolates bilinearly in position and
        # linearly in angle in a single call -- rather than sampling all K slices
        # and gathering two of them, which costs K times as much in the inner
        # loop.  The volume holds per-cell minima (see the class docstring), so
        # its samples live at cell centres rather than on the nodes.
        self._r, self._rk = self._as_cell_volumes(ray, ray_kind)

    @staticmethod
    def _as_cell_volumes(ray, ray_kind):
        """(K,ny,nx) nodal fields -> (1,1,K,ny-1,nx-1) per-cell minima + their kinds.

        Cell (k, j, i) spans directions k..k+1 (wrapping, so the last cell spans
        K-1..0) and nodes j..j+1 / i..i+1, and holds the smallest of the eight
        nodal distances at its corners, together with the boundary kind that went
        with that smallest one.
        """
        if ray is None:
            return None, None
        K, ny, nx = ray.shape
        if ny < 2 or nx < 2:
            raise ValueError("ray field needs at least a 2x2 grid")
        wrap = torch.cat([ray, ray[:1]], dim=0)                       # (K+1,ny,nx)
        corners = torch.stack([wrap[dk:dk + K, dj:dj + ny - 1, di:di + nx - 1]
                               for dk in (0, 1) for dj in (0, 1) for di in (0, 1)], dim=0)
        vals, idx = corners.min(dim=0)                                # (K,ny-1,nx-1)
        rv = vals.unsqueeze(0).unsqueeze(0)
        if ray_kind is None:
            return rv, None
        wrapk = torch.cat([ray_kind, ray_kind[:1]], dim=0)
        kcorners = torch.stack([wrapk[dk:dk + K, dj:dj + ny - 1, di:di + nx - 1]
                                for dk in (0, 1) for dj in (0, 1) for di in (0, 1)], dim=0)
        kv = kcorners.gather(0, idx.unsqueeze(0)).squeeze(0)
        return rv, kv.unsqueeze(0).unsqueeze(0)

    @property
    def device(self):
        return self.grid.device

    @property
    def dtype(self):
        return self.grid.dtype

    @property
    def has_ray_field(self):
        return self.ray is not None

    @property
    def n_dirs(self):
        return 0 if self.ray is None else int(self.ray.shape[0])

    def to(self, device=None, dtype=None):
        g = self.grid.to(device=device, dtype=dtype)
        r = self.ray.to(device=device, dtype=dtype) if self.ray is not None else None
        rk = self.ray_kind.to(device=device, dtype=dtype) if self.ray_kind is not None else None
        return RegionFields(g, self.minx, self.miny, self.maxx, self.maxy, ray=r, ray_kind=rk)

    # -- shared affine ------------------------------------------------------

    def _norm_coords(self, x, y):
        gx = 2.0 * (x - self.minx) / (self.maxx - self.minx) - 1.0
        gy = 2.0 * (y - self.miny) / (self.maxy - self.miny) - 1.0
        return gx, gy

    def _norm_coords_ray_cell(self, x, y):
        """Normalized coords into the ray field's *cell-centre* lattice.

        The pooled volume has one entry per cell, so its samples sit half a cell
        inside the bbox on each side; `align_corners=True` then maps -1/+1 onto
        the first/last cell centre.
        """
        _, ny, nx = self.ray.shape
        hx = (self.maxx - self.minx) / (nx - 1)
        hy = (self.maxy - self.miny) / (ny - 1)
        lo_x, hi_x = self.minx + 0.5 * hx, self.maxx - 0.5 * hx
        lo_y, hi_y = self.miny + 0.5 * hy, self.maxy - 0.5 * hy
        # a 2-node axis collapses to a single cell centre; keep the divide finite
        sx = max(hi_x - lo_x, 1e-12)
        sy = max(hi_y - lo_y, 1e-12)
        gx = 2.0 * (x - lo_x) / sx - 1.0
        gy = 2.0 * (y - lo_y) / sy - 1.0
        return gx, gy

    def sample(self, points):
        """points: (..., 2) tensor of world coordinates. Returns (...,) signed distance.

        Differentiable. Inside the grid bbox this is bilinear interpolation of the
        rasterized field. Outside it, the clamped border value (always negative,
        since the grid is padded with a margin beyond the region) minus the
        euclidean distance to the bbox -- so far-away points still get a gradient
        pointing back toward the region instead of the flat zero that a
        zero-padded `grid_sample` would give.
        """
        pts = points.reshape(-1, 2)
        x, y = pts[:, 0], pts[:, 1]

        gx, gy = self._norm_coords(x, y)

        ox = torch.clamp(self.minx - x, min=0.0) + torch.clamp(x - self.maxx, min=0.0)
        oy = torch.clamp(self.miny - y, min=0.0) + torch.clamp(y - self.maxy, min=0.0)
        d_out = torch.sqrt(ox * ox + oy * oy + 1e-18)

        g = torch.stack([gx.clamp(-1.0, 1.0), gy.clamp(-1.0, 1.0)], dim=-1).view(1, 1, -1, 2)
        vals = F.grid_sample(
            self._g.to(dtype=pts.dtype), g.to(dtype=pts.dtype),
            mode="bilinear", padding_mode="border", align_corners=True,
        )[0, 0, 0]
        return (vals - d_out).reshape(points.shape[:-1])

    # -- directional (ray) distance ----------------------------------------

    def ray_distance(self, points, angles, with_kind=False):
        """Distance from each point to the material boundary along `angles`.

        points: (..., 2) world coordinates. angles: (...) radians (0 = +X, CCW).
        Returns (...,) distances, trilinearly interpolated over the *per-cell
        minimum* volume described in the class docstring -- bilinear in position,
        linear in angle (wrapping at 2*pi), and conservative, i.e. it under- but
        never over-reports the clear distance around the query.  Differentiable
        in both position and angle.

        With `with_kind=True` also returns a (...,) tensor of 0.0 (the nearest
        crossing was the part's outer edge) / 1.0 (it was an interior hole or
        keepout), sampled nearest-neighbour and detached -- it is a label, not a
        quantity to differentiate.
        """
        if self.ray is None:
            raise RuntimeError("ray field has not been built for this RegionFields")
        shape = points.shape[:-1]
        pts = points.reshape(-1, 2)
        ang = angles.reshape(-1)
        x, y = pts[:, 0], pts[:, 1]
        gx, gy = self._norm_coords_ray_cell(x, y)

        # Depth coordinate: the K angular cells are centred on (k + 1/2) * 2pi/K,
        # so -1 lands on cell 0 and +1 on cell K-1.  An angle in the outer half
        # cell (within pi/K of 0 or 2pi) clamps onto the first/last cell, which is
        # exactly the cell that spans the wrap.
        K = self.ray.shape[0]
        hz = 2.0 * math.pi / K
        th = torch.remainder(ang, 2.0 * math.pi)
        gz = 2.0 * (th - 0.5 * hz) / max(2.0 * math.pi - hz, 1e-12) - 1.0

        g = torch.stack([gx.clamp(-1.0, 1.0), gy.clamp(-1.0, 1.0), gz.clamp(-1.0, 1.0)], dim=-1) \
                 .view(1, 1, 1, -1, 3)
        out = F.grid_sample(
            self._r.to(dtype=pts.dtype), g.to(dtype=pts.dtype),
            mode="bilinear", padding_mode="border", align_corners=True,
        )[0, 0, 0, 0]

        if not with_kind:
            return out.reshape(shape)

        with torch.no_grad():
            kind = F.grid_sample(
                self._rk.to(dtype=pts.dtype), g.to(dtype=pts.dtype),
                mode="nearest", padding_mode="border", align_corners=True,
            )[0, 0, 0, 0]
        return out.reshape(shape), kind.reshape(shape)


# Back-compat alias: the class used to only hold the SDF.
RegionSDF = RegionFields


def _grid_dims(minx, miny, maxx, maxy, resolution):
    w, h = maxx - minx, maxy - miny
    if w >= h:
        nx = int(resolution)
        ny = max(2, int(round(resolution * h / w)))
    else:
        ny = int(resolution)
        nx = max(2, int(round(resolution * w / h)))
    return nx, ny


def boundary_segments(region):
    """Every boundary segment of `region` as (A (S,2), B (S,2), kind (S,) uint8).

    kind is 0 for a segment of an exterior ring (the part's outer edge) and 1 for
    an interior ring (a hole or keepout wall).
    """
    geoms = list(region.geoms) if isinstance(region, MultiPolygon) else [region]
    A, B, K = [], [], []
    for g in geoms:
        if g.is_empty:
            continue
        rings = [(np.asarray(g.exterior.coords), 0)]
        rings += [(np.asarray(r.coords), 1) for r in g.interiors]
        for coords, kind in rings:
            if len(coords) < 2:
                continue
            A.append(coords[:-1, :2])
            B.append(coords[1:, :2])
            K.append(np.full(len(coords) - 1, kind, dtype=np.uint8))
    if not A:
        return (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, dtype=np.uint8))
    return np.concatenate(A), np.concatenate(B), np.concatenate(K)


def _ray_cast_grid(px, py, segA, segB, segKind, angles, chunk=4096):
    """Analytic ray cast: distance from each (px, py) along each angle to the
    nearest segment crossing.

    Returns (dist (K, M), kind (K, M)).  A shapely LineString-per-cell
    intersection would be the obvious implementation but costs a C call per
    (cell, direction) pair; this is the same computation done as a handful of
    numpy array ops, which is ~30x faster at the grid sizes used here.
    """
    M = px.shape[0]
    K = angles.shape[0]
    dist = np.zeros((K, M), dtype=np.float64)
    kind = np.zeros((K, M), dtype=np.uint8)
    if segA.shape[0] == 0 or M == 0:
        return dist, kind

    # float32 throughout: the field is later sampled bilinearly on a grid whose
    # pitch is ~1 drawing unit, so 1e-5 relative precision is far below the
    # discretization error and costs half the memory traffic of float64.
    f = np.float32
    ax, ay = segA[:, 0].astype(f), segA[:, 1].astype(f)
    bx, by = segB[:, 0].astype(f), segB[:, 1].astype(f)
    dx, dy = bx - ax, by - ay
    c_ad = ax * dy - ay * dx                             # A x d, per segment
    pxf, pyf = px.astype(f), py.astype(f)
    INF = f(np.inf)

    for k in range(K):
        ux, uy = f(math.cos(float(angles[k]))), f(math.sin(float(angles[k])))
        den = ux * dy - uy * dx                          # u x d, per segment

        # The ray's infinite line crosses segment AB exactly when the endpoints
        # straddle it, i.e. when `u x (A-P)` and `u x (B-P)` differ in sign.
        # Both of those are an outer difference of a per-segment constant and a
        # per-point constant, so the whole test reduces to "is kp between ka and
        # kb" -- two bool comparisons instead of solving for the segment
        # parameter.  Half-open on one side so a ray through a shared vertex is
        # counted once.
        ka = ux * ay - uy * ax
        kb = ux * by - uy * bx
        klo = np.minimum(ka, kb)
        khi = np.maximum(ka, kb)

        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            X = pxf[s:e, None]
            Y = pyf[s:e, None]
            kp = ux * Y - uy * X                         # u x P
            with np.errstate(divide="ignore", invalid="ignore"):
                # segments parallel to u give den == 0 -> inf/nan, which the
                # straddle mask below discards anyway
                t = (c_ad[None, :] - (X * dy[None, :] - Y * dx[None, :])) / den[None, :]
            np.copyto(t, INF, where=~((kp > klo[None, :]) & (kp <= khi[None, :])))
            np.copyto(t, INF, where=~(t >= 0.0))         # also catches NaN
            j = np.argmin(t, axis=1)
            best = t[np.arange(e - s), j]
            good = np.isfinite(best)
            dist[k, s:e] = np.where(good, best, 0.0)
            kind[k, s:e] = np.where(good, segKind[j], 0)

    return dist, kind


def build_ray_field(material_region, minx, miny, maxx, maxy, n_dirs=32,
                    resolution=128, dtype=torch.float64, device=None, cap=None):
    """Rasterize the directional distance field of `material_region`.

    Returns (ray (K, ny, nx) tensor, kind (K, ny, nx) tensor, (nx, ny)).
    Cells outside the material get *minus* their distance back to the boundary,
    which makes `lc` negative there and so drives the bearing factor to its floor
    -- exactly the barrier behaviour wanted for a bolt that has wandered out of
    the part, and unlike a flat 0 it still has a gradient pointing back inside.

    `cap` clips the distance from above.  The bearing model saturates at a clear
    distance of 2d, so anything past ~2.5d is indistinguishable to it, and
    clipping keeps the field's dynamic range small: the residual interpolation
    error where a ray grazes past a hole is then bounded by the cap instead of by
    the width of the plate.
    """
    nx, ny = _grid_dims(minx, miny, maxx, maxy, resolution)
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(xs, ys)
    fx, fy = X.ravel(), Y.ravel()

    # Cost is O(cells x segments x directions), and a DXF's tessellated circles
    # contribute most of the segments. Simplifying by a tolerance far below the
    # grid pitch cuts the segment count several-fold with no visible effect on
    # the field.
    # The tolerance has to stay well under the grid pitch *and* under the
    # smallest feature the model cares about: a simplified arc that cuts a
    # millimetre off a small hole is a millimetre off every clear distance that
    # hole governs.
    span = max(maxx - minx, maxy - miny)
    simp_tol = min(span / 2000.0, span / max(resolution, 2) / 20.0)
    cast_region = material_region
    try:
        s = material_region.simplify(simp_tol, preserve_topology=True)
        if not s.is_empty and s.is_valid:
            cast_region = s
    except Exception:
        pass

    segA, segB, segKind = boundary_segments(cast_region)
    angles = np.arange(n_dirs) * (2.0 * math.pi / n_dirs)

    inside = shapely.contains_xy(material_region, fx, fy)
    idx = np.flatnonzero(inside)

    dist = np.zeros((n_dirs, fx.shape[0]), dtype=np.float64)
    kind = np.zeros((n_dirs, fx.shape[0]), dtype=np.uint8)
    if idx.size:
        d_in, k_in = _ray_cast_grid(fx[idx], fy[idx], segA, segB, segKind, angles)
        if cap is not None:
            d_in = np.minimum(d_in, float(cap))
        dist[:, idx] = d_in
        kind[:, idx] = k_in

    out = np.flatnonzero(~inside)
    if out.size:
        # Outside the material the "clear distance" is meaningless; store the
        # signed continuation -(distance back to the boundary) so the field falls
        # smoothly through zero instead of stepping onto a flat plateau that the
        # min-pool would then smear a cell deep into the part.
        back = shapely.distance(material_region.boundary,
                                shapely.points(fx[out], fy[out]))
        if cap is not None:
            back = np.minimum(back, float(cap))
        dist[:, out] = -back[None, :]

    ray = torch.as_tensor(dist.reshape(n_dirs, ny, nx), dtype=dtype)
    rkind = torch.as_tensor(kind.reshape(n_dirs, ny, nx).astype(np.float64), dtype=dtype)
    if device is not None:
        ray, rkind = ray.to(device), rkind.to(device)
    return ray, rkind, (nx, ny)


def build_region_sdf(region, resolution=384, margin_frac=0.05, device=None,
                     dtype=torch.float64, material_region=None, n_dirs=0,
                     ray_resolution=256, ray_cap=None):
    """Rasterize `region` (Polygon or MultiPolygon) into a `RegionFields`.

    resolution: number of grid samples along the longer bbox side.
    margin_frac: bbox padding as a fraction of the longer side (the border of the
      grid must lie outside the region for the out-of-grid fallback to be valid).
    material_region: the unshrunk part material (outline minus keepouts/holes).
      When given it sets the grid bbox (it always contains `region`) so the SDF
      and the ray field share one affine mapping, and with `n_dirs > 0` the
      directional distance field of the bearing model is built against it.
    ray_cap: clip the ray field at this distance (see `build_ray_field`).
    """
    if region.is_empty:
        raise ValueError("cannot build an SDF for an empty region")

    if material_region is not None and not material_region.is_empty:
        bminx, bminy, bmaxx, bmaxy = material_region.bounds
        rminx, rminy, rmaxx, rmaxy = region.bounds
        minx, miny = min(bminx, rminx), min(bminy, rminy)
        maxx, maxy = max(bmaxx, rmaxx), max(bmaxy, rmaxy)
    else:
        minx, miny, maxx, maxy = region.bounds
    span = max(maxx - minx, maxy - miny)
    margin = max(span * margin_frac, span / max(resolution, 2) * 2.0, 1e-9)
    minx, miny, maxx, maxy = minx - margin, miny - margin, maxx + margin, maxy + margin

    nx, ny = _grid_dims(minx, miny, maxx, maxy, resolution)

    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)
    fx, fy = X.ravel(), Y.ravel()

    # shapely 2.x vectorized array API -- one C call for all cells, no python loop.
    pts = shapely.points(fx, fy)
    dist = shapely.distance(region.boundary, pts)
    inside = shapely.contains_xy(region, fx, fy)
    sdf = np.where(inside, dist, -dist).reshape(ny, nx)

    grid = torch.as_tensor(sdf, dtype=dtype)
    if device is not None:
        grid = grid.to(device)

    ray = rkind = None
    if n_dirs and material_region is not None and not material_region.is_empty:
        ray, rkind, _ = build_ray_field(
            material_region, minx, miny, maxx, maxy, n_dirs=n_dirs,
            resolution=ray_resolution, dtype=dtype, device=device, cap=ray_cap,
        )

    return RegionFields(grid, minx, miny, maxx, maxy, ray=ray, ray_kind=rkind)
