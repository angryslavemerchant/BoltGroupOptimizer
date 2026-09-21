"""FastAPI backend: in-memory single-project state for the bolt group optimizer."""
import asyncio
import contextlib
import io
import os
import tempfile
import math
from typing import Optional, List

import ezdxf
import numpy as np
import shapely
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask

from app.dxf_loader import load_dxf
from app.geometry import (build_legal_region, build_material_region,
                          region_to_geojson_like, region_tolerance, build_region_sdf)
from app.optimizer import run_optimization, score_layouts

app = FastAPI()

STATE = {
    "dxf_path": None,
    "loops": [],
    "open_chains": [],
    "snap_points": [],
    "units": "unknown",
    "bbox": None,
    "roles": {},  # loop_index -> "part" | "keepout" | "ignore"
    "datum": None,  # (x, y)
    # Independent load cases: each has its own application point and vector, and
    # is evaluated on its own against the same bolt group (never superposed).
    "forces": [],  # [{name, x, y, fx, fy}]
    # User-placed bolts. Full members of the bolt group -- they carry load and
    # count toward the centroid/polar moment -- but the optimizer never moves them.
    "fixed_bolts": [],  # [{x, y}]
    "settings": {
        "edge_clearance": 10.0,
        "min_spacing": 20.0,
        "load_ceiling": 1000.0,
        "n_min": 2,
        "n_max": 12,
        "seeds_per_n": 80,
        "iterations": 300,
        "use_gpu": False,
        # Bearing / tear-out model (AISC J3.10 shaped). `bolt_diameter` is the
        # hole diameter lc is measured from; capacity saturates at lc = 2d.
        "bolt_diameter": 6.0,
        "bearing_enabled": True,
        "k_min": 0.05,
        "edge_weight": 0.0,
        "units_override": None,
    },
    # The layout currently on screen, whatever produced it:
    #   {"source": "optimized"|"alternative"|"manual",
    #    "result": <scored layout payload>,
    #    "derived_from": {"result", "source"} | None}
    # /result and /export.dxf read this; "derived_from" is what Revert restores.
    "layout": None,
    "last_result": None,  # back-compat mirror of STATE["layout"]["result"]
    "alternatives": [],   # top-K distinct layouts for the winning N
    # Cached geometry, keyed on the inputs that define it -- so a drag can
    # re-score a layout without rebuilding the (expensive) rasterized fields.
    "_cache": {},
}


def part_loop_index():
    for idx, role in STATE["roles"].items():
        if role == "part":
            return idx
    return None


def keepout_loop_indices():
    return [idx for idx, role in STATE["roles"].items() if role == "keepout"]


def ignore_loop_indices():
    return [idx for idx, role in STATE["roles"].items() if role == "ignore"]


@app.get("/")
def index():
    # no-cache: without it Chrome heuristically caches the page (no Cache-Control
    # header is sent otherwise) and keeps serving a stale frontend after edits.
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "..", "static", "index.html"),
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/load-dxf")
async def load_dxf_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        data = load_dxf(tmp_path)
    except Exception as e:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"Failed to parse DXF: {e}")

    # the previous upload's temp copy is no longer referenced by anything
    old = STATE.get("dxf_path")
    if old and old != tmp_path:
        with contextlib.suppress(OSError):
            os.remove(old)
    STATE["dxf_path"] = tmp_path
    STATE["loops"] = data["loops"]
    STATE["open_chains"] = data["open_chains"]
    STATE["snap_points"] = data["snap_points"]
    STATE["units"] = data["units"]
    STATE["bbox"] = data["bbox"]
    STATE["roles"] = {}
    STATE["datum"] = None
    STATE["forces"] = []
    STATE["fixed_bolts"] = []
    STATE["layout"] = None
    STATE["last_result"] = None
    STATE["alternatives"] = []
    STATE["_cache"] = {}

    return {
        "loops": data["loops"],
        "open_chains": data["open_chains"],
        "snap_points": data["snap_points"],
        "units": data["units"],
        "bbox": data["bbox"],
        "message": data.get("message"),
        "notes": data.get("notes") or [],
        "ignored": data.get("ignored") or {},
        "paperspace_entities": data.get("paperspace_entities", 0),
    }


class SelectBody(BaseModel):
    loop_index: int
    role: str  # "part" | "keepout" | "ignore" | "none"


@app.post("/select")
def select(body: SelectBody):
    if body.loop_index < 0 or body.loop_index >= len(STATE["loops"]):
        raise HTTPException(status_code=400, detail="loop_index out of range")
    if body.role == "none":
        STATE["roles"].pop(body.loop_index, None)
    elif body.role == "part":
        # clear any other part assignment
        for idx in list(STATE["roles"].keys()):
            if STATE["roles"][idx] == "part":
                del STATE["roles"][idx]
        STATE["roles"][body.loop_index] = "part"
    elif body.role == "keepout":
        STATE["roles"][body.loop_index] = "keepout"
    elif body.role == "ignore":
        STATE["roles"][body.loop_index] = "ignore"
    else:
        raise HTTPException(status_code=400, detail="role must be part, keepout, ignore, or none")
    return {"roles": STATE["roles"]}


class DatumBody(BaseModel):
    x: float
    y: float


@app.post("/datum")
def set_datum(body: DatumBody):
    STATE["datum"] = (body.x, body.y)
    return {"datum": STATE["datum"]}


class ForceCase(BaseModel):
    name: Optional[str] = None
    x: float
    y: float
    fx: float
    fy: float


class ForcesBody(BaseModel):
    forces: List[ForceCase]


def _normalized_forces(cases):
    return [{"name": c.name or f"Case {i + 1}", "x": c.x, "y": c.y, "fx": c.fx, "fy": c.fy}
            for i, c in enumerate(cases)]


@app.post("/forces")
def set_forces(body: ForcesBody):
    """Replace the whole list of load cases. Simplest thing the UI needs: it always
    holds the full list client-side and re-posts it after any edit."""
    STATE["forces"] = _normalized_forces(body.forces)
    return {"forces": STATE["forces"]}


@app.get("/forces")
def get_forces():
    return {"forces": STATE["forces"]}


class FixedBolt(BaseModel):
    x: float
    y: float


class FixedBoltsBody(BaseModel):
    bolts: List[FixedBolt]


_region_tolerance = region_tolerance


@app.post("/fixed-bolts")
def set_fixed_bolts(body: FixedBoltsBody):
    """Replace the whole list of user-placed bolts (the UI owns the list).

    Every bolt is validated against the *current* legal region, so raising the
    edge clearance after placing bolts surfaces here as a named error rather
    than as a silently-infeasible optimization run.
    """
    bolts = [{"x": b.x, "y": b.y} for b in body.bolts]
    if bolts:
        # validate against the region *these* bolts would produce (a bolt in an
        # existing hole makes that hole occupied rather than removed material)
        region = _compute_region(bolts)
        if region.is_empty:
            raise HTTPException(
                status_code=400,
                detail="Legal placement region is empty — cannot place fixed bolts.")
        check = region.buffer(_region_tolerance(region))
        for i, b in enumerate(bolts):
            if not shapely.contains_xy(check, b["x"], b["y"]):
                raise HTTPException(
                    status_code=400,
                    detail=(f"Fixed bolt {i + 1} at ({b['x']:.3f}, {b['y']:.3f}) is "
                            f"outside the legal placement region (it may be inside a "
                            f"keepout or closer than the {STATE['settings']['edge_clearance']} "
                            f"edge clearance). Move or remove it, or loosen the clearance."),
                )
    STATE["fixed_bolts"] = bolts
    return {"fixed_bolts": STATE["fixed_bolts"]}


@app.get("/fixed-bolts")
def get_fixed_bolts():
    return {"fixed_bolts": STATE["fixed_bolts"]}


class SettingsBody(BaseModel):
    edge_clearance: Optional[float] = None
    min_spacing: Optional[float] = None
    load_ceiling: Optional[float] = None
    n_min: Optional[int] = None
    n_max: Optional[int] = None
    seeds_per_n: Optional[int] = None
    iterations: Optional[int] = None
    use_gpu: Optional[bool] = None
    bolt_diameter: Optional[float] = None
    bearing_enabled: Optional[bool] = None
    k_min: Optional[float] = None
    edge_weight: Optional[float] = None
    units_override: Optional[str] = None


@app.post("/settings")
def set_settings(body: SettingsBody):
    for k, v in body.model_dump(exclude_unset=True).items():
        STATE["settings"][k] = v
    return {"settings": STATE["settings"]}


@app.get("/settings")
def get_settings():
    return {"settings": STATE["settings"]}


def _occupied_hole_indices(fixed_bolts=None):
    """Indices of the existing holes (closed loops inside the part that are not
    keepouts) that have a fixed bolt sitting in them.

    Every closed loop inside the part is treated as removed material, so a hole
    centre -- the place a user-placed bolt normally goes -- would always be
    illegal and would always read as a material edge for tear-out.  A hole with
    a bolt in it is occupied, not empty: it is dropped from the hole set so the
    bolt is legal there and bears against the plate around it.  A loop the user
    marked as a keepout is never occupied this way.
    """
    bolts = STATE["fixed_bolts"] if fixed_bolts is None else fixed_bolts
    if not bolts:
        return set()
    part_idx = part_loop_index()
    keepouts = set(keepout_loop_indices())
    ignores = set(ignore_loop_indices())
    out = set()
    for i, loop in enumerate(STATE["loops"]):
        if i == part_idx or i in keepouts or i in ignores or len(loop) < 3:
            continue
        poly = shapely.Polygon(loop)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        if any(shapely.contains_xy(poly, b["x"], b["y"]) for b in bolts):
            out.add(i)
    return out


def _region_inputs(fixed_bolts=None):
    """(part_loop, keepout_loops, hole_loops).

    `hole_loops` -- the real holes, subtracted from the *material* region and
    therefore a bearing edge -- excludes the part loop, occupied holes, every
    keepout (solid material, placement-only) and every loop marked ignore
    (construction geometry, no geometric effect at all).
    """
    part_idx = part_loop_index()
    if part_idx is None:
        raise HTTPException(status_code=400, detail="No part loop selected.")
    part_loop = STATE["loops"][part_idx]
    keepout_idx = set(keepout_loop_indices())
    ignore_idx = set(ignore_loop_indices())
    keepout_loops = [STATE["loops"][i] for i in keepout_idx]
    occupied = _occupied_hole_indices(fixed_bolts)
    hole_loops = [loop for i, loop in enumerate(STATE["loops"])
                  if i != part_idx and i not in occupied
                  and i not in keepout_idx and i not in ignore_idx]
    return part_loop, keepout_loops, hole_loops


def _compute_region(fixed_bolts=None):
    part_loop, keepout_loops, hole_loops = _region_inputs(fixed_bolts)
    return build_legal_region(part_loop, keepout_loops,
                              STATE["settings"]["edge_clearance"], other_loops=hole_loops)


def _compute_material_region():
    """Unshrunk placement region -- the real material edge the bearing model uses.

    Keepouts are solid material here (not subtracted): a keepout forbids
    placement but is not a bearing/tear-out edge.
    """
    part_loop, _keepout_loops, hole_loops = _region_inputs()
    return build_material_region(part_loop, [], other_loops=hole_loops)


# ---------------------------------------------------------------------------
# Cached geometry for interactive scoring
#
# POST /evaluate runs on every frame of a bolt drag, so it must not rebuild the
# legal region, the material region or (above all) the rasterized RegionFields,
# whose ray field costs seconds.  All three are cached under a key naming every
# input that defines them; anything that changes the geometry changes the key,
# so there is no invalidation to remember to do by hand.
# ---------------------------------------------------------------------------

def _geometry_key():
    s = STATE["settings"]
    return (
        STATE["dxf_path"], part_loop_index(), tuple(sorted(keepout_loop_indices())),
        tuple(sorted(ignore_loop_indices())),
        tuple(sorted(_occupied_hole_indices())),
        float(s.get("edge_clearance") or 0.0),
        int(s.get("sdf_resolution", 384) or 384),
        float(s.get("bolt_diameter") or 0.0),
        bool(s.get("bearing_enabled", True)),
        int(s.get("ray_dirs", 32) or 32),
        int(s.get("ray_resolution", 256) or 256),
        float(s.get("k_min") or 0.05),
    )


def _cached_geometry():
    """(region, material_region, region_check, bearing) with everything memoized.

    `region_check` is the tol-buffered legal region, prepared for fast repeated
    point-in-polygon tests; `bearing` is the dict `score_layouts` wants, or None
    when the bearing model is off or unusable.
    """
    key = _geometry_key()
    cache = STATE["_cache"]
    if cache.get("key") == key:
        return cache["region"], cache["material"], cache["check"], cache["bearing"]

    s = STATE["settings"]
    region = _compute_region()
    if region.is_empty:
        raise HTTPException(status_code=400, detail="Legal placement region is empty.")
    material = _compute_material_region()
    check = region.buffer(region_tolerance(region))
    shapely.prepare(check)

    d = float(s.get("bolt_diameter") or 0.0)
    bearing_on = (bool(s.get("bearing_enabled", True)) and d > 0.0
                  and material is not None and not material.is_empty)
    fields = build_region_sdf(
        region, resolution=int(s.get("sdf_resolution", 384) or 384),
        material_region=material,
        n_dirs=int(s.get("ray_dirs", 32) or 32) if bearing_on else 0,
        ray_resolution=int(s.get("ray_resolution", 256) or 256),
        ray_cap=(3.0 * d) if bearing_on else None,
    )
    bearing = ({"fields": fields, "d": d, "k_min": float(s.get("k_min") or 0.05)}
               if (bearing_on and fields.has_ray_field) else None)

    STATE["_cache"] = {"key": key, "region": region, "material": material,
                       "check": check, "fields": fields, "bearing": bearing}
    return region, material, check, bearing


def _evaluate_bolts(bolts):
    """Score one hand-made layout with the optimizer's own scoring path.

    `bolts` is [{x, y, fixed}] in world coordinates. The `fixed` flag is a label
    only -- it changes nothing in the physics (it is what the optimizer is not
    allowed to move, and nothing is being optimized here) -- so the user's bolt
    order is preserved rather than shuffling the pinned ones to the front.
    """
    if not bolts:
        raise HTTPException(status_code=400, detail="No bolts to evaluate.")
    if not STATE["forces"]:
        raise HTTPException(status_code=400, detail="No force case set.")

    region, _material, check, bearing = _cached_geometry()

    P = np.array([[float(b["x"]), float(b["y"])] for b in bolts], dtype=float)[None, :, :]
    M = np.ones((1, P.shape[1]), dtype=bool)
    in_region = shapely.contains_xy(check, P[0, :, 0], P[0, :, 1]).reshape(1, -1)

    cases = STATE["forces"]
    layout = score_layouts(
        P, M,
        [[c["x"], c["y"]] for c in cases],
        [[c["fx"], c["fy"]] for c in cases],
        [c["name"] for c in cases],
        bearing=bearing,
        min_spacing=float(STATE["settings"]["min_spacing"]),
        fixed_count=0,
        region_ok=in_region.all(axis=1),
        per_bolt_region_ok=in_region,
    )[0]

    flags = [bool(b.get("fixed")) for b in bolts]
    layout["fixed_flags"] = flags
    for row, f in zip(layout["per_bolt"], flags):
        row["fixed"] = f
    return layout


def _rescore_current():
    """Re-score the displayed layout against the *current* inputs.

    A result is a set of positions; the numbers attached to it (loads, lc, k,
    feasibility) are only valid for the forces, clearance, spacing, diameter
    and part/keepout selection that were in force when it was scored.  Rather
    than remember to invalidate it on every settings edit, `/result`,
    `/layout/revert` and `/export.dxf` re-score the positions on the way out,
    so what the user reads and exports always reflects the sidebar as it is now.
    Positions themselves are untouched.  Raises the usual 400s when the layout
    can no longer be scored (no part loop, no force case, empty region).
    """
    layout = STATE.get("layout")
    if not layout:
        return None
    res = layout["result"]
    flags = res.get("fixed_flags") or []
    bolts = [{"x": p[0], "y": p[1], "fixed": bool(flags[i]) if i < len(flags) else False}
             for i, p in enumerate(res["positions"])]
    new = _evaluate_bolts(bolts)
    layout["result"] = new
    STATE["last_result"] = new
    return new


@app.get("/region")
def get_region():
    region = _compute_region()
    polys = region_to_geojson_like(region)
    if not polys:
        return {"polygons": [], "empty": True}
    return {"polygons": polys, "empty": False}


@app.websocket("/optimize")
async def optimize_ws(ws: WebSocket):
    await ws.accept()

    # Cooperative cancellation: a stop request (explicit {"cmd": "stop"} message,
    # or the socket simply closing e.g. the browser tab going away) sets this flag,
    # which run_optimization polls between restarts/iterations. Runs concurrently
    # with the send loop below since the client won't send anything else mid-run.
    stop_flag = {"stop": False}

    async def receive_loop():
        try:
            while True:
                msg = await ws.receive_json()
                if isinstance(msg, dict) and msg.get("cmd") == "stop":
                    stop_flag["stop"] = True
        except Exception:
            # Disconnect, non-JSON message, or socket already closed by the send
            # side finishing normally -- either way, stop the optimization loop.
            stop_flag["stop"] = True

    receiver_task = asyncio.create_task(receive_loop())

    try:
        if part_loop_index() is None:
            await ws.send_json({"status": "error", "message": "No part loop selected."})
            await ws.close()
            return
        if STATE["datum"] is None:
            await ws.send_json({"status": "error", "message": "No datum point set."})
            await ws.close()
            return
        if not STATE["forces"]:
            await ws.send_json({"status": "error", "message": "No force case set."})
            await ws.close()
            return

        # The rasterized fields take seconds to build on a detailed part; tell
        # the client, then build them off the event loop so the frame actually
        # goes out first.  The same cached fields then serve the drag-evaluate
        # loop, so a run never rasterizes twice.
        await ws.send_json({"status": "building", "message": "Building geometry fields…"})
        try:
            region, material_region, check, _bearing = await asyncio.to_thread(_cached_geometry)
        except HTTPException as e:
            await ws.send_json({"status": "error", "message": e.detail})
            await ws.close()
            return
        fields = STATE["_cache"].get("fields")

        # Re-validate here as well as in POST /fixed-bolts: the clearance can have
        # been raised after the bolts were accepted, which shrinks the region
        # under them.
        bad = [i + 1 for i, b in enumerate(STATE["fixed_bolts"])
               if not shapely.contains_xy(check, b["x"], b["y"])]
        if bad:
            await ws.send_json({
                "status": "error",
                "message": (f"Fixed bolt(s) {', '.join(map(str, bad))} are outside the "
                            f"legal placement region — move them or loosen the edge "
                            f"clearance."),
                "bad_fixed_bolts": [i - 1 for i in bad],
            })
            await ws.close()
            return

        async for frame in run_optimization(
            region, STATE["forces"], STATE["settings"],
            should_stop=lambda: stop_flag["stop"],
            material_region=material_region,
            fixed_bolts=STATE["fixed_bolts"],
            fields=fields,
        ):
            await ws.send_json(frame)
            if frame.get("status") == "done":
                if frame.get("result"):
                    _set_layout(frame["result"], "optimized")
                STATE["alternatives"] = frame.get("alternatives") or []
        await ws.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"status": "error", "message": str(e)})
            await ws.close()
        except Exception:
            pass
    finally:
        receiver_task.cancel()
        # asyncio.CancelledError is a BaseException (not Exception) since Python
        # 3.8, so it must be named explicitly here or awaiting the cancelled
        # task logs an "Exception in ASGI application" traceback on every
        # normal completion, even though nothing actually went wrong.
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await receiver_task


@app.get("/device")
def get_device():
    """Frontend uses this to decide whether the "use GPU" checkbox means anything."""
    import torch
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


class AlternativeBody(BaseModel):
    index: int


@app.post("/select-alternative")
def select_alternative(body: AlternativeBody):
    """Make one of the reported alternative layouts the current one.

    /result and /export.dxf both read STATE["layout"], so swapping it here is
    all that is needed for the results table and the DXF export to follow whichever
    layout the user is looking at.
    """
    alts = STATE.get("alternatives") or []
    if body.index < 0 or body.index >= len(alts):
        raise HTTPException(status_code=400, detail="alternative index out of range")
    alt = alts[body.index]
    _set_layout({
        "n": alt["n"],
        "positions": alt["positions"],
        "peak_load": alt["peak_load"],
        "raw_peak_load": alt.get("raw_peak_load"),
        "per_bolt_loads": alt.get("per_bolt_loads", []),
        "per_bolt": alt.get("per_bolt", []),
        "fixed_flags": alt.get("fixed_flags", []),
        "bearing": alt.get("bearing", False),
        "per_case": alt.get("per_case", []),
        "governing_case": alt.get("governing_case"),
        "feasible": alt.get("feasible", True),
    }, "alternative")
    return {"selected": body.index, "n": alt["n"], "peak_load": alt["peak_load"]}


# ---------------------------------------------------------------------------
# The displayed layout
# ---------------------------------------------------------------------------

def _set_layout(result, source, derived_from=None):
    STATE["layout"] = {"source": source, "result": result, "derived_from": derived_from}
    # Mirror, so anything still reading STATE["last_result"] sees the same layout.
    STATE["last_result"] = result
    return STATE["layout"]


def _current_result():
    layout = STATE.get("layout")
    return layout["result"] if layout else None


def _layout_feasibility(result):
    """(feasible, message). A layout is exportable only when every bolt is inside
    the legal region and no pair is closer than the minimum spacing."""
    rows = result.get("per_bolt") or []
    outside = [i + 1 for i, r in enumerate(rows) if r.get("in_region") is False]
    crowded = [i + 1 for i, r in enumerate(rows) if r.get("spacing_ok") is False]
    if outside:
        return False, (f"Bolt(s) {', '.join(map(str, outside))} are outside the legal "
                       f"placement region (inside a keepout, or closer to an edge than "
                       f"the {STATE['settings']['edge_clearance']} edge clearance). "
                       f"Move them back in, or Revert.")
    if crowded:
        return False, (f"Bolt(s) {', '.join(map(str, crowded))} are closer than the "
                       f"{STATE['settings']['min_spacing']} minimum bolt spacing. "
                       f"Move them apart, or Revert.")
    return True, ""


def _result_payload(result):
    """The response shape /result, /evaluate, /layout and /layout/revert all share,
    so the frontend renders a dragged layout with the same code as an optimized one.

    Without a datum the coordinates are reported absolute (`datum` is null), so a
    fixed-only layout can be dragged and scored before the datum is picked."""
    datum = STATE["datum"]
    dx, dy = datum if datum else (0.0, 0.0)
    flags = result.get("fixed_flags") or []
    rows = result.get("per_bolt") or []
    bolts_rel = []
    for i, p in enumerate(result["positions"]):
        row = rows[i] if i < len(rows) else {}
        bolts_rel.append({
            "x": p[0] - dx, "y": p[1] - dy,
            "fixed": bool(flags[i]) if i < len(flags) else bool(row.get("fixed")),
            "in_region": bool(row.get("in_region", True)),
            "spacing_ok": bool(row.get("spacing_ok", True)),
        })
    # One table row per load case: name, application point relative to datum, vector.
    forces_rel = [{"name": f["name"], "x": f["x"] - dx, "y": f["y"] - dy,
                   "fx": f["fx"], "fy": f["fy"]} for f in STATE["forces"]]
    ceiling = float(STATE["settings"]["load_ceiling"])
    feasible, reason = _layout_feasibility(result)
    layout = STATE.get("layout") or {}
    return {
        "datum": {"x": dx, "y": dy} if datum else None,
        "bolts": bolts_rel,
        "forces": forces_rel,
        "units": STATE["settings"]["units_override"] or STATE["units"],
        "peak_load": result["peak_load"],
        "raw_peak_load": result.get("raw_peak_load"),
        "peak_eff": result["peak_load"],
        "peak_raw": result.get("raw_peak_load"),
        "n": result["n"],
        "per_bolt": rows,
        "bearing": bool(result.get("bearing")),
        "bolt_diameter": STATE["settings"].get("bolt_diameter"),
        "per_case": result.get("per_case", []),
        "governing_case": result.get("governing_case"),
        "fixed_bolts": STATE["fixed_bolts"],
        "positions": result["positions"],
        "source": layout.get("source"),
        "can_revert": bool(layout.get("derived_from")),
        "derived_from_source": (layout.get("derived_from") or {}).get("source"),
        "feasible": feasible,
        "infeasible_reason": reason,
        "ceiling": ceiling,
        "ceiling_met": bool(result["peak_load"] <= ceiling),
    }


class EvalBolt(BaseModel):
    x: float
    y: float
    fixed: bool = False


class EvaluateBody(BaseModel):
    bolts: List[EvalBolt]


@app.post("/evaluate")
def evaluate(body: EvaluateBody):
    """Score one arbitrary bolt layout without touching the displayed one.

    This is the live feedback behind a bolt drag: the frontend posts the bolt
    positions it is currently showing, ~30 times a second, and gets back exactly
    the numbers the results table already displays.  It never stores anything --
    POST /layout does that, once, when the drag ends.
    """
    result = _evaluate_bolts([{"x": b.x, "y": b.y, "fixed": b.fixed} for b in body.bolts])
    return _result_payload(result)


class LayoutBody(BaseModel):
    bolts: List[EvalBolt]
    source: Optional[str] = "manual"


@app.post("/layout")
def set_layout(body: LayoutBody):
    """Make an arbitrary bolt layout the displayed one (a drag has just ended).

    A manual layout remembers what it was edited from, so Revert can put the
    optimizer's answer back; editing an already-manual layout keeps the original
    ancestor rather than chaining, so Revert always means "back to the result".
    """
    result = _evaluate_bolts([{"x": b.x, "y": b.y, "fixed": b.fixed} for b in body.bolts])
    source = body.source or "manual"
    prev = STATE.get("layout")
    derived = None
    if source == "manual" and prev:
        derived = (prev.get("derived_from") if prev.get("source") == "manual"
                   else {"result": prev["result"], "source": prev["source"]})
    _set_layout(result, source, derived)
    return _result_payload(result)


@app.post("/layout/revert")
def revert_layout():
    layout = STATE.get("layout")
    derived = layout.get("derived_from") if layout else None
    if not derived:
        raise HTTPException(status_code=400, detail="Nothing to revert to.")
    _set_layout(derived["result"], derived.get("source", "optimized"))
    return _result_payload(_rescore_current())


@app.get("/result")
def get_result():
    if _current_result() is None:
        raise HTTPException(status_code=400, detail="No optimization result yet.")
    return _result_payload(_rescore_current())


def _add_note_block(doc, msp, lines, x, y, height, layer):
    """Multi-line note at (x, y): MTEXT where the DXF version allows it, stacked
    TEXT entities on R12 (which has no MTEXT and would otherwise refuse the save)."""
    if doc.dxfversion >= "AC1015":
        mtext = msp.add_mtext("\\P".join(lines),
                              dxfattribs={"layer": layer, "char_height": height})
        mtext.set_location(insert=(x, y))
        return
    for i, line in enumerate(lines):
        msp.add_text(line, dxfattribs={"layer": layer, "height": height}) \
           .set_placement((x, y - i * height * 1.6))


@app.get("/export.dxf")
def export_dxf():
    if _current_result() is None:
        raise HTTPException(status_code=400, detail="No optimization result yet.")
    if STATE["dxf_path"] is None or not os.path.exists(STATE["dxf_path"]):
        raise HTTPException(status_code=400, detail="No source DXF loaded.")
    # score the positions against the inputs as they are *now*, never a stale
    # breakdown from before a settings edit
    result = _rescore_current()
    # An edited layout can be illegal; refuse rather than exporting a drawing
    # whose bolts sit in a keepout or on top of each other.
    ok, why = _layout_feasibility(result)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Cannot export: {why}")

    doc = ezdxf.readfile(STATE["dxf_path"])
    msp = doc.modelspace()

    for layer, color in [("BOLTS", 1), ("FIXED_BOLTS", 4), ("DATUM", 3), ("FORCE", 5)]:
        if layer not in doc.layers:
            doc.layers.add(name=layer, color=color)

    dx, dy = STATE["datum"] if STATE["datum"] else (0.0, 0.0)
    # Draw bolts at their real hole size when a diameter is set, so the export
    # is dimensionally meaningful rather than a symbol.
    bolt_d = float(STATE["settings"].get("bolt_diameter") or 0.0)
    bolt_radius = bolt_d / 2.0 if bolt_d > 0 else max(STATE["settings"]["min_spacing"] * 0.15, 1.0)
    cross_size = max(STATE["settings"]["min_spacing"] * 0.15, 1.0) * 2

    # User-placed bolts go on their own layer and carry a cross through the hole,
    # so an optimized bolt and a pinned one are distinguishable in the DXF.
    fixed_flags = result.get("fixed_flags") or []
    for i, p in enumerate(result["positions"]):
        is_fixed = bool(fixed_flags[i]) if i < len(fixed_flags) else False
        layer = "FIXED_BOLTS" if is_fixed else "BOLTS"
        msp.add_circle(center=(p[0], p[1]), radius=bolt_radius, dxfattribs={"layer": layer})
        if is_fixed:
            r = bolt_radius
            msp.add_line((p[0] - r, p[1]), (p[0] + r, p[1]), dxfattribs={"layer": layer})
            msp.add_line((p[0], p[1] - r), (p[0], p[1] + r), dxfattribs={"layer": layer})

    # Datum: crosshair plus longer +X / +Y stubs with labels, so the sign
    # convention of every "(rel datum)" coordinate below is unambiguous.
    msp.add_line((dx - cross_size, dy), (dx + cross_size, dy), dxfattribs={"layer": "DATUM"})
    msp.add_line((dx, dy - cross_size), (dx, dy + cross_size), dxfattribs={"layer": "DATUM"})
    axis = cross_size * 3
    msp.add_line((dx, dy), (dx + axis, dy), dxfattribs={"layer": "DATUM"})
    msp.add_line((dx, dy), (dx, dy + axis), dxfattribs={"layer": "DATUM"})
    small = cross_size * 0.6
    msp.add_text("+X", dxfattribs={"layer": "DATUM", "height": small}).set_placement(
        (dx + axis + small * 0.5, dy - small * 0.5))
    msp.add_text("+Y", dxfattribs={"layer": "DATUM", "height": small}).set_placement(
        (dx - small * 0.5, dy + axis + small * 0.5))
    msp.add_text("DATUM", dxfattribs={"layer": "DATUM", "height": cross_size}).set_placement(
        (dx + cross_size, dy + cross_size)
    )

    # One arrow per load case, labelled with the case name.
    for case in STATE["forces"]:
        fx0, fy0 = case["x"], case["y"]
        fvx, fvy = case["fx"], case["fy"]
        mag = math.hypot(fvx, fvy) or 1.0
        scale = cross_size * 3 / mag
        fx1, fy1 = fx0 + fvx * scale, fy0 + fvy * scale
        msp.add_line((fx0, fy0), (fx1, fy1), dxfattribs={"layer": "FORCE"})
        msp.add_text(case["name"], dxfattribs={"layer": "FORCE", "height": cross_size * 0.8}) \
           .set_placement((fx1, fy1))

    per_case = {p["name"]: p for p in (result.get("per_case") or [])}
    bearing_on = bool(result.get("bearing"))
    units = STATE['settings']['units_override'] or STATE['units']
    lines = [
        f"DATUM: ({dx:.3f}, {dy:.3f}) in drawing coords"
        + ("" if STATE["datum"] else " (no datum picked: drawing origin used)")
        + "; +X right, +Y up; all coordinates below are relative to it",
        f"UNITS: {units} (lengths); loads in the units the force magnitudes were entered in",
    ]
    if bearing_on:
        raw = result.get("raw_peak_load")
        lines.append(f"N BOLTS: {result['n']}   PEAK EFF LOAD: {result['peak_load']:.3f}"
                     + (f"   (RAW {raw:.3f})" if raw is not None else ""))
        lines.append(
            f"BEARING/TEAR-OUT: bolt dia d = {bolt_d:.3f}; lc = clear distance from "
            f"hole edge to nearest material edge (outer profile, hole or adjacent "
            f"bolt) along that bolt's load direction;"
        )
        lines.append(
            f"  k = clamp(lc / 2d, {float(STATE['settings'].get('k_min') or 0.05):.2f}, 1) "
            f"(capacity saturates at lc = 2d = {2*bolt_d:.3f}); "
            f"EFF LOAD = LOAD / k, compared against the ceiling."
        )
    else:
        lines.append(f"N BOLTS: {result['n']}   PEAK LOAD: {result['peak_load']:.3f}")
        lines.append("BEARING/TEAR-OUT MODEL: off (edge clearance only)")
    if result.get("governing_case"):
        lines.append(f"GOVERNING CASE: {result['governing_case']}")
    if any(fixed_flags):
        lines.append(
            f"FIXED (USER-PLACED) BOLTS: {sum(1 for f in fixed_flags if f)} of "
            f"{len(result['positions'])}, marked (fixed) below and drawn on layer "
            f"FIXED_BOLTS")
    for case in STATE["forces"]:
        peak = per_case.get(case["name"], {}).get("peak")
        peak_txt = f"   PEAK: {peak:.3f}" if peak is not None else ""
        lines.append(
            f"FORCE {case['name']}: PT (rel datum) "
            f"({case['x']-dx:.3f}, {case['y']-dy:.3f})   "
            f"VEC ({case['fx']:.3f}, {case['fy']:.3f}){peak_txt}"
        )
    per_bolt = result.get("per_bolt") or []
    if bearing_on and per_bolt:
        lines.append("BOLT  X (rel datum)  Y (rel datum)  LOAD  LC  K  EFF LOAD  LIMITER")
    for i, p in enumerate(result["positions"]):
        tag = " (fixed)" if (i < len(fixed_flags) and fixed_flags[i]) else ""
        base = f"BOLT {i+1}{tag} (rel datum): ({p[0]-dx:.3f}, {p[1]-dy:.3f})"
        if bearing_on and i < len(per_bolt):
            pb = per_bolt[i]
            k = float(pb.get('k', 0.0))
            # the limiter only means something while it actually limits
            limiter = "none (lc >= 2d, full capacity)" if k >= 1.0 else pb.get('limiter', '')
            base += (f"   LOAD {pb.get('load', 0.0):.3f}   LC {pb.get('lc', 0.0):.3f}"
                     f"   K {k:.3f}   EFF {pb.get('effective_load', 0.0):.3f}"
                     f"   LIMITED BY {limiter}")
        lines.append(base)

    text_height = cross_size * 0.8
    minx = STATE["bbox"]["minx"] if STATE["bbox"] else dx
    maxy = STATE["bbox"]["maxy"] if STATE["bbox"] else dy
    table_x = minx
    table_y = maxy + cross_size * 4
    _add_note_block(doc, msp, lines, table_x, table_y, text_height, "DATUM")

    out_path = tempfile.NamedTemporaryFile(suffix=".dxf", delete=False).name
    doc.saveas(out_path)
    return FileResponse(out_path, filename="bolt_layout.dxf", media_type="application/dxf",
                        background=BackgroundTask(_remove_quietly, out_path))


def _remove_quietly(path):
    with contextlib.suppress(OSError):
        os.remove(path)


static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
