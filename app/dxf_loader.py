"""Read a DXF, flatten curved entities, and chain segments into closed loops."""
import math
import ezdxf
import shapely
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, polygonize_full

INSUNITS_NAMES = {
    0: "unitless", 1: "in", 2: "ft", 3: "mi", 4: "mm", 5: "cm", 6: "m",
    7: "km", 8: "microinches", 9: "mils", 10: "yd", 13: "um", 14: "dm",
}

ARC_SEGMENTS = 32  # flattening resolution for arcs/circles
SPLINE_SEGMENTS = 64

PRIMITIVE_TYPES = ("LINE", "CIRCLE", "ARC", "LWPOLYLINE", "POLYLINE", "SPLINE", "ELLIPSE")


def _iter_primitives(entities, depth=0, max_depth=64, ignored=None):
    """Recursively walk entities, expanding INSERTs into world-space copies.

    ezdxf's `insert.virtual_entities()` returns transformed (translated,
    rotated, non-uniformly scaled, mirrored) copies of a block's contents
    in WCS, already flattened one level. Nested INSERTs among those virtual
    copies still need expanding themselves, hence the recursion. `max_depth`
    guards against a pathological/cyclic block reference blowing the stack.

    `ignored` (a dict) collects a count per entity type that is skipped, so a
    drawing made of HATCH / TEXT / DIMENSION entities is reported as such
    rather than silently loading as empty.
    """
    if depth > max_depth:
        return
    for e in entities:
        dxftype = e.dxftype()
        if dxftype == "INSERT":
            try:
                virtual = list(e.virtual_entities())
            except Exception:
                continue
            yield from _iter_primitives(virtual, depth + 1, max_depth, ignored)
        elif dxftype in PRIMITIVE_TYPES:
            yield e
        elif ignored is not None:
            ignored[dxftype] = ignored.get(dxftype, 0) + 1


def _arc_points(center, radius, start_deg, end_deg, segments=ARC_SEGMENTS):
    start = math.radians(start_deg)
    end = math.radians(end_deg)
    if end <= start:
        end += 2 * math.pi
    pts = []
    for i in range(segments + 1):
        t = start + (end - start) * i / segments
        pts.append((center[0] + radius * math.cos(t), center[1] + radius * math.sin(t)))
    return pts


def _circle_points(center, radius, segments=ARC_SEGMENTS * 2):
    pts = []
    for i in range(segments):
        t = 2 * math.pi * i / segments
        pts.append((center[0] + radius * math.cos(t), center[1] + radius * math.sin(t)))
    pts.append(pts[0])
    return pts


class Segment:
    """A single open polyline chain candidate, with an entity source tag."""
    def __init__(self, points, closed=False):
        self.points = points  # list of (x, y)
        self.closed = closed


def _entity_to_segments(e):
    """Return a list of Segment for one DXF entity (open chains, not yet merged)."""
    dxftype = e.dxftype()
    segs = []
    if dxftype == "LINE":
        p1 = (e.dxf.start.x, e.dxf.start.y)
        p2 = (e.dxf.end.x, e.dxf.end.y)
        segs.append(Segment([p1, p2]))
    elif dxftype == "CIRCLE":
        c = (e.dxf.center.x, e.dxf.center.y)
        pts = _circle_points(c, e.dxf.radius)
        segs.append(Segment(pts, closed=True))
    elif dxftype == "ARC":
        c = (e.dxf.center.x, e.dxf.center.y)
        pts = _arc_points(c, e.dxf.radius, e.dxf.start_angle, e.dxf.end_angle)
        segs.append(Segment(pts))
    elif dxftype in ("LWPOLYLINE", "POLYLINE"):
        try:
            points_raw = list(e.vertices()) if dxftype == "POLYLINE" else None
        except Exception:
            points_raw = None
        pts = []
        if dxftype == "LWPOLYLINE":
            pts_with_bulge = [(p[0], p[1], p[4] if len(p) > 4 else 0.0) for p in e.get_points()]
            for i in range(len(pts_with_bulge)):
                x, y, bulge = pts_with_bulge[i]
                pts.append((x, y))
                if bulge and i < len(pts_with_bulge) - 1:
                    x2, y2, _ = pts_with_bulge[i + 1]
                    pts.extend(_bulge_points((x, y), (x2, y2), bulge)[1:-1])
                elif bulge and e.closed:
                    x2, y2, _ = pts_with_bulge[0]
                    pts.extend(_bulge_points((x, y), (x2, y2), bulge)[1:-1])
            closed = bool(e.closed)
            if closed and pts and pts[0] != pts[-1]:
                pts.append(pts[0])
        else:  # POLYLINE (3D/2D legacy)
            for v in e.vertices:
                loc = v.dxf.location
                pts.append((loc.x, loc.y))
            closed = bool(e.is_closed)
            if closed and pts and pts[0] != pts[-1]:
                pts.append(pts[0])
        segs.append(Segment(pts, closed=closed))
    elif dxftype == "SPLINE":
        try:
            flat = e.flattening(distance=None, segments=SPLINE_SEGMENTS)
            pts = [(p.x, p.y) for p in flat]
        except Exception:
            try:
                pts = [(c.x, c.y) for c in e.control_points]
            except Exception:
                pts = []
        closed = e.closed if hasattr(e, "closed") else (len(pts) > 1 and pts[0] == pts[-1])
        if closed and pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        segs.append(Segment(pts, closed=closed))
    elif dxftype == "ELLIPSE":
        # A CIRCLE/ARC inside a block gets non-uniformly scaled by an INSERT's
        # virtual_entities() into an ELLIPSE (ezdxf can't represent a scaled
        # circle as a CIRCLE anymore). Flatten it the same way as a SPLINE.
        try:
            flat = e.flattening(distance=None, segments=ARC_SEGMENTS * 2)
            pts = [(p.x, p.y) for p in flat]
        except Exception:
            pts = []
        start = getattr(e.dxf, "start_param", 0.0)
        end = getattr(e.dxf, "end_param", 2 * math.pi)
        full_circle = abs((end - start) % (2 * math.pi)) < 1e-9
        closed = full_circle and len(pts) > 1
        if closed and pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        segs.append(Segment(pts, closed=closed))
    return segs


def _bulge_points(p1, p2, bulge, segments=16):
    """Flatten an LWPOLYLINE bulge (arc) between two vertices into points."""
    x1, y1 = p1
    x2, y2 = p2
    chord = math.hypot(x2 - x1, y2 - y1)
    if chord < 1e-12:
        return [p1, p2]
    angle = 4 * math.atan(bulge)
    radius = chord / (2 * math.sin(angle / 2)) if abs(math.sin(angle / 2)) > 1e-12 else 1e9
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    d = math.sqrt(max(radius ** 2 - (chord / 2) ** 2, 0.0))
    # perpendicular direction, sign depends on bulge sign
    dx, dy = (x2 - x1) / chord, (y2 - y1) / chord
    nx, ny = -dy, dx
    sign = 1 if bulge > 0 else -1
    cx = mx + nx * d * sign
    cy = my + ny * d * sign
    start_ang = math.atan2(y1 - cy, x1 - cx)
    end_ang = start_ang + angle
    pts = []
    for i in range(segments + 1):
        t = start_ang + (end_ang - start_ang) * i / segments
        pts.append((cx + radius * math.cos(t), cy + radius * math.sin(t)))
    return pts


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ring_signature(coords, grid=1e-6):
    """Order/direction/start-independent key for a closed ring.

    `polygonize_full` returns every *minimal face* of the noded arrangement, so a
    ring that bounds a hole shows up twice: once as the interior ring of the face
    around it, and once as the exterior ring of the face inside it. Emitting both
    gives the UI two copies of the same loop, which an even-odd fill then cancels
    back to "filled" (two crossings instead of one) -- i.e. holes that look solid.
    Keyed on the rounded vertex set (minus the repeated closing point), the two
    copies collapse to one.
    """
    pts = list(coords)
    if len(pts) > 1 and _dist(pts[0], pts[-1]) <= grid:
        pts = pts[:-1]
    q = sorted((round(p[0] / grid), round(p[1] / grid)) for p in pts)
    return tuple(q)


def _dedupe_rings(loops, grid=1e-6):
    """Drop rings that repeat an earlier ring's vertex set (see _ring_signature)."""
    out, seen = [], set()
    for loop in loops:
        sig = _ring_signature(loop, grid=grid)
        if not sig or sig in seen:
            continue
        seen.add(sig)
        out.append(loop)
    return out


def _dedupe_consecutive(points):
    """Drop consecutive duplicate points so a LineString stays valid."""
    if not points:
        return points
    cleaned = [points[0]]
    for p in points[1:]:
        if p != cleaned[-1]:
            cleaned.append(p)
    return cleaned


def _chain_segments(open_segments, gap_tol):
    """Chain open point-chains end-to-end within gap_tol using shapely.

    Endpoints are snapped onto a precision grid of size gap_tol so that
    small real-world gaps (export noise, duplicate/overlapping lines) close
    up, then the segments are noded and merged with unary_union. Polygonize
    turns any resulting rings into closed loops (interior rings become their
    own loops); anything left dangling or unresolved is reported as an open
    chain. This scales to tens of thousands of segments, unlike the old
    O(n^2) greedy chaining scan.

    Returns (closed_loops, open_chains) as lists of point-tuple lists.
    """
    lines = []
    for seg in open_segments:
        pts = _dedupe_consecutive(list(seg.points))
        if len(pts) < 2:
            continue
        lines.append(LineString(pts))

    if not lines:
        return [], []

    merged = MultiLineString(lines)
    grid = max(gap_tol, 1e-9)
    try:
        merged = shapely.set_precision(merged, grid_size=grid, mode="valid_output")
    except Exception:
        pass

    if merged.is_empty:
        return [], []

    unioned = unary_union(merged)
    polygons, dangles, cuts, invalid = polygonize_full(unioned)

    closed_loops = []
    for poly in polygons.geoms:
        closed_loops.append(list(poly.exterior.coords))
        for interior in poly.interiors:
            closed_loops.append(list(interior.coords))

    open_chains = []
    for collection in (dangles, cuts, invalid):
        for geom in collection.geoms:
            coords = list(geom.coords)
            if len(coords) >= 2:
                open_chains.append(coords)

    return _dedupe_rings(closed_loops), open_chains


def load_dxf(path, gap_tol=0.01):
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    if "$INSUNITS" in doc.header:
        units = INSUNITS_NAMES.get(doc.header.get("$INSUNITS"), "unknown")
    else:
        # $INSUNITS absent entirely: the drawing never declared units at all,
        # which is different from an explicit INSUNITS=0 ("unitless"). Report
        # "unknown" so the UI prompts for a units override instead of
        # silently assuming a unit.
        units = "unknown"

    closed_loops = []
    open_segments = []
    snap_points = []
    entities_read = 0
    ignored = {}

    for e in _iter_primitives(msp, ignored=ignored):
        dxftype = e.dxftype()
        entities_read += 1
        try:
            segs = _entity_to_segments(e)
        except Exception:
            continue

        if dxftype in ("CIRCLE", "ARC", "ELLIPSE"):
            c = (e.dxf.center.x, e.dxf.center.y)
            snap_points.append({"x": c[0], "y": c[1], "kind": "center"})

        for seg in segs:
            if not seg.points or len(seg.points) < 2:
                continue
            for p in seg.points:
                snap_points.append({"x": p[0], "y": p[1], "kind": "vertex"})
            if seg.closed:
                closed_loops.append(seg.points)
            else:
                open_segments.append(seg)

    chained_loops, open_chains = _chain_segments(open_segments, gap_tol)
    closed_loops.extend(chained_loops)
    # A ring can also be duplicated across sources (e.g. a CIRCLE drawn twice, or
    # a circle that is also traced by loose arcs), so dedupe the combined list.
    # The chainer snapped its output onto a `gap_tol` grid while entity rings
    # keep their raw coordinates, so the two copies only match at that grid --
    # a 1e-6 grid here let every arc-traced circle through twice.
    closed_loops = _dedupe_rings(closed_loops, grid=max(gap_tol, 1e-9))

    # dedupe near-identical snap points
    dedup = []
    seen = set()
    for p in snap_points:
        key = (round(p["x"], 6), round(p["y"], 6), p["kind"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)

    bbox = None
    all_pts = [p for loop in closed_loops for p in loop] + [p for chain in open_chains for p in chain]
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        bbox = {"minx": min(xs), "miny": min(ys), "maxx": max(xs), "maxy": max(ys)}

    # Only model space is read. A drawing whose geometry lives in a paper-space
    # layout would otherwise load as silently empty.
    paperspace_entities = 0
    try:
        for layout in doc.layouts:
            if not layout.is_modelspace:
                paperspace_entities += len(layout)
    except Exception:
        pass

    notes = []
    if ignored:
        notes.append("ignored " + ", ".join(f"{n}×{t}" for t, n in sorted(ignored.items())))
    if paperspace_entities:
        notes.append(f"{paperspace_entities} entit{'ies' if paperspace_entities != 1 else 'y'} "
                     f"in paper space (not read)")

    message = None
    if not closed_loops:
        message = (
            f"No closed loops found — {len(open_chains)} open chain"
            f"{'s' if len(open_chains) != 1 else ''}, {entities_read} entit"
            f"{'ies' if entities_read != 1 else 'y'} read."
        )
        if notes:
            message += " (" + "; ".join(notes) + ")"

    return {
        "loops": [[list(p) for p in loop] for loop in closed_loops],
        "open_chains": [[list(p) for p in chain] for chain in open_chains],
        "snap_points": dedup,
        "units": units,
        "bbox": bbox,
        "message": message,
        "ignored": ignored,
        "paperspace_entities": paperspace_entities,
        "notes": notes,
    }
