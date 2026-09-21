import os
import sys
import time
import asyncio
import math

import numpy as np
import pytest
import torch
from shapely.geometry import Point, Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.geometry import build_legal_region, build_region_sdf
from app.optimizer import (
    run_optimization, build_seeds, batched_bolt_loads, batched_loss,
    peak_loads, farthest_point_subset, sample_pool,
    bearing_clear_distance, effective_loads, LIMITER_BOLT, LIMITER_HOLE,
)


RECT = [(0, 0), (200, 0), (200, 100), (0, 100), (0, 0)]


def _case(point, vector, name="Case 1"):
    return {"name": name, "x": point[0], "y": point[1], "fx": vector[0], "fy": vector[1]}


def _run(region, force_point, force_vector, settings, stream_every=1000):
    """Single-case convenience wrapper around the multi-case API."""
    return _run_cases(region, [_case(force_point, force_vector)], settings, stream_every)


def _run_cases(region, cases, settings, stream_every=1000):
    async def go():
        return [f async for f in run_optimization(
            region, cases, settings, stream_every=stream_every)]
    return asyncio.run(go())


def test_region_shrinks_with_edge_clearance():
    region_0 = build_legal_region(RECT, [], 0.0)
    region_10 = build_legal_region(RECT, [], 10.0)
    assert region_10.area < region_0.area
    minx, miny, maxx, maxy = region_10.bounds
    assert minx >= 10 - 1e-6
    assert miny >= 10 - 1e-6
    assert maxx <= 190 + 1e-6
    assert maxy <= 90 + 1e-6


# --------------------------------------------------------------------------
# SDF
# --------------------------------------------------------------------------

def test_sdf_sign_and_distance_on_rect_with_hole():
    outer = [(0, 0), (200, 0), (200, 100), (0, 100)]
    hole = [(90, 40), (110, 40), (110, 60), (90, 60)]
    region = Polygon(outer, [hole])
    sdf = build_region_sdf(region, resolution=512)

    pts = torch.tensor([
        [100.0, 50.0],   # centre of the hole -> outside the region
        [40.0, 50.0],    # well inside material, 40 from the left edge
        [-20.0, 50.0],   # outside the outer boundary, 20 away
        [5.0, 50.0],     # 5 from the left edge
        [100.0, 20.0],   # 20 below the hole
    ], dtype=torch.float64)
    v = sdf.sample(pts)

    assert v[0] < 0            # hole interior is not legal
    assert v[1] > 0            # material interior is legal
    assert v[2] < 0            # off-part is not legal

    # known distances (grid pitch is ~0.4 units here, bilinear -> ~1 cell error)
    assert abs(abs(v[0].item()) - 10.0) < 1.0     # hole centre is 10 from hole wall
    assert abs(v[1].item() - 40.0) < 1.0
    assert abs(v[3].item() - 5.0) < 1.0
    assert abs(v[4].item() - 20.0) < 1.0
    assert abs(abs(v[2].item()) - 20.0) < 1.0     # out-of-grid fallback term


def test_sdf_is_differentiable_and_points_inward():
    region = build_legal_region(RECT, [], 10.0)
    sdf = build_region_sdf(region, resolution=256)
    p = torch.tensor([[195.0, 50.0]], dtype=torch.float64, requires_grad=True)
    sdf.sample(p).sum().backward()
    # increasing sdf means moving away from the near (right-hand) edge -> -x
    assert p.grad[0, 0] < 0


# --------------------------------------------------------------------------
# Batched physics vs. a straightforward unbatched implementation
# --------------------------------------------------------------------------

def _reference_loads(positions, force_point, force_vector):
    """Plain, unvectorized elastic/polar method -- the thing the batched code must match."""
    n = len(positions)
    cx = sum(p[0] for p in positions) / n
    cy = sum(p[1] for p in positions) / n
    J = sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in positions)
    moment = (force_point[0] - cx) * force_vector[1] - (force_point[1] - cy) * force_vector[0]
    mags = []
    for p in positions:
        rx, ry = p[0] - cx, p[1] - cy
        fx = force_vector[0] / n + (moment / J) * (-ry)
        fy = force_vector[1] / n + (moment / J) * (rx)
        mags.append(math.hypot(fx, fy))
    return mags


# Accuracy tolerances are a property of the *dtype*, not of the assertion: the
# search runs in float32 on the DirectML backend (which has no float64 kernels),
# so the same identities have to hold there, just to ~7 significant figures
# instead of ~16. Parametrizing keeps the float64 bar exactly where it was
# rather than weakening one tolerance to cover both.
DTYPE_TOL = {
    torch.float64: dict(load=1e-6, peak=1e-9, loss=1e-6),
    torch.float32: dict(load=2e-3, peak=1e-4, loss=2e-3),
}
DTYPES = list(DTYPE_TOL)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).replace("torch.", ""))
def test_batched_matches_unbatched_including_masked_slots(dtype):
    tol = DTYPE_TOL[dtype]
    pts = [(20.0, 30.0), (150.0, 25.0), (120.0, 80.0)]
    fp, fv = (60.0, 90.0), (300.0, -700.0)
    ref = _reference_loads(pts, fp, fv)
    scale = max(ref)

    # batch of one, with two dead slots padded on the end
    P = torch.tensor([[*pts, (0.0, 0.0), (999.0, -999.0)]], dtype=dtype)
    M = torch.tensor([[True, True, True, False, False]])
    _, mags = batched_bolt_loads(P, M,
                                 torch.tensor(fp, dtype=dtype),
                                 torch.tensor(fv, dtype=dtype))
    assert mags.shape == (1, 1, 5)   # (batch, case, slot)
    got = mags[0, 0, :3].tolist()
    for a, b in zip(got, ref):
        assert abs(a - b) < tol["load"] * max(scale, 1.0)
    # dead slots contribute nothing
    assert mags[0, 0, 3].item() == 0.0 and mags[0, 0, 4].item() == 0.0

    soft, hard = peak_loads(mags, M)
    assert abs(hard[0].item() - max(ref)) < tol["peak"] * max(scale, 1.0)
    assert soft[0].item() >= hard[0].item() - tol["peak"] * max(scale, 1.0)

    # and the full loss reduces to the unbatched formula when no constraint bites
    region = build_legal_region(RECT, [], 10.0)
    sdf = build_region_sdf(region, resolution=256, dtype=dtype)
    loss, info = batched_loss(P, M, torch.tensor(fp, dtype=dtype),
                              torch.tensor(fv, dtype=dtype), sdf, min_spacing=5.0,
                              weights={"region": 0.0, "spacing": 0.0})
    assert abs(info["hard"][0].item() - max(ref)) < tol["loss"] * max(scale, 1.0)
    assert abs(loss[0].item() - soft[0].item()) < tol["loss"] * max(scale, 1.0)


def test_batched_rows_are_independent():
    """Stacking two different layouts must give each the same answer it gets alone."""
    a = [(20.0, 30.0), (150.0, 25.0), (120.0, 80.0)]
    b = [(30.0, 30.0), (170.0, 70.0)]
    fp = torch.tensor([60.0, 90.0], dtype=torch.float64)
    fv = torch.tensor([300.0, -700.0], dtype=torch.float64)
    P = torch.tensor([[*a], [*b, (0.0, 0.0)]], dtype=torch.float64)
    M = torch.tensor([[True, True, True], [True, True, False]])
    _, mags = batched_bolt_loads(P, M, fp, fv)
    assert np.allclose(mags[0, 0].tolist(), _reference_loads(a, (60.0, 90.0), (300.0, -700.0)), atol=1e-9)
    assert np.allclose(mags[1, 0, :2].tolist(), _reference_loads(b, (60.0, 90.0), (300.0, -700.0)), atol=1e-9)


# --------------------------------------------------------------------------
# Multiple independent load cases
# --------------------------------------------------------------------------

def test_two_identical_cases_reproduce_the_single_case_result():
    """Duplicating a case must change nothing: each case is evaluated on its own,
    so a second identical one adds no load and must not perturb the search."""
    region = build_legal_region(RECT, [], edge_clearance=10.0)
    settings = {
        "min_spacing": 15.0, "load_ceiling": 400.0,
        "n_min": 2, "n_max": 5, "seeds_per_n": 20, "iterations": 60,
        "edge_clearance": 10.0,
        "seed": 7,
    }
    one = _run_cases(region, [_case((100.0, 50.0), (0.0, -1000.0), "A")], settings)
    two = _run_cases(region, [_case((100.0, 50.0), (0.0, -1000.0), "A"),
                              _case((100.0, 50.0), (0.0, -1000.0), "B")], settings)
    r1 = [f for f in one if f["status"] == "done"][0]["result"]
    r2 = [f for f in two if f["status"] == "done"][0]["result"]
    assert r1["n"] == r2["n"]
    assert abs(r1["peak_load"] - r2["peak_load"]) < 1e-9
    assert np.allclose(np.array(r1["positions"]), np.array(r2["positions"]), atol=1e-9)
    assert [c["peak"] for c in r2["per_case"]] == [r2["per_case"][0]["peak"]] * 2
    assert abs(r2["per_case"][0]["peak"] - r2["peak_load"]) < 1e-9


def test_two_different_cases_peak_is_max_of_independent_evaluations():
    """The reported hard peak must equal the worse of the two cases, each recomputed
    with the plain unbatched elastic method -- i.e. cases are independent, not summed."""
    region = build_legal_region(RECT, [], edge_clearance=10.0)
    cases = [
        _case((40.0, 90.0), (0.0, -1200.0), "Extended"),
        _case((170.0, 20.0), (800.0, 300.0), "Closed"),
    ]
    settings = {
        "min_spacing": 15.0, "load_ceiling": 500.0,
        "n_min": 3, "n_max": 5, "seeds_per_n": 20, "iterations": 80,
        "edge_clearance": 10.0,
        "seed": 3,
    }
    final = [f for f in _run_cases(region, cases, settings) if f["status"] == "done"][0]
    result = final["result"]
    pts = [tuple(p) for p in result["positions"]]

    per_case_ref = []
    for c in cases:
        mags = _reference_loads(pts, (c["x"], c["y"]), (c["fx"], c["fy"]))
        per_case_ref.append(max(mags))

    assert abs(result["peak_load"] - max(per_case_ref)) < 1e-6
    got = {p["name"]: p["peak"] for p in result["per_case"]}
    assert abs(got["Extended"] - per_case_ref[0]) < 1e-6
    assert abs(got["Closed"] - per_case_ref[1]) < 1e-6
    assert result["governing_case"] == cases[int(np.argmax(per_case_ref))]["name"]
    # the governing bolt really is the worst-loaded one in its case
    for i, c in enumerate(cases):
        mags = _reference_loads(pts, (c["x"], c["y"]), (c["fx"], c["fy"]))
        idx = result["per_case"][i]["governing_bolt_index"]
        assert abs(mags[idx] - max(mags)) < 1e-6
    # and every alternative carries the same breakdown
    for alt in final["alternatives"]:
        assert len(alt["per_case"]) == 2
        assert abs(alt["peak_load"] - max(p["peak"] for p in alt["per_case"])) < 1e-9


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

def test_seeds_are_inside_region_and_span_every_n():
    region = build_legal_region(RECT, [], edge_clearance=10.0)
    rng = np.random.default_rng(1)
    P, M, ns = build_seeds(region, 2, 6, 20, min_spacing=15.0, rng=rng)
    assert set(ns.tolist()) == {2, 3, 4, 5, 6}
    buffered = region.buffer(1e-6)
    for b in range(len(P)):
        for j in np.flatnonzero(M[b]):
            assert buffered.contains(Point(P[b, j, 0], P[b, j, 1]))
        assert int(M[b].sum()) == ns[b]


def test_farthest_point_seeds_are_well_spread():
    region = build_legal_region(RECT, [], edge_clearance=10.0)
    rng = np.random.default_rng(2)
    pool = sample_pool(region, 400, rng)
    for _ in range(10):
        pts = farthest_point_subset(pool, 4, rng)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        d[np.diag_indices(4)] = np.inf
        # greedy FPS on this region should beat the min-spacing default comfortably
        assert d.min() > 20.0


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------

def test_optimizer_result_is_legal_and_meets_ceiling():
    region = build_legal_region(RECT, [], edge_clearance=10.0)
    settings = {
        "min_spacing": 15.0, "load_ceiling": 5000.0,
        "n_min": 2, "n_max": 6, "seeds_per_n": 20, "iterations": 60,
        "edge_clearance": 10.0,
    }
    frames = _run(region, (100.0, 50.0), (0.0, -1000.0), settings)
    done = [f for f in frames if f["status"] == "done"]
    assert len(done) == 1
    final = done[0]
    result = final["result"]
    assert result["peak_load"] <= settings["load_ceiling"]
    assert final["ceiling_met"] is True

    pts = np.array(result["positions"])
    assert len(pts) == result["n"]
    buffered = region.buffer(1e-6)
    for x, y in pts:
        assert buffered.contains(Point(x, y))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    d[np.diag_indices(len(pts))] = np.inf
    assert d.min() >= settings["min_spacing"] - 1e-6


def test_result_is_best_of_all_reported_alternatives():
    """The returned layout must be at least as good as every alternative reported
    for the same N -- i.e. the winner really is the batch's best, not just one of them."""
    region = build_legal_region(RECT, [], edge_clearance=10.0)
    settings = {
        "min_spacing": 15.0, "load_ceiling": 5000.0,
        "n_min": 3, "n_max": 3, "seeds_per_n": 30, "iterations": 40,
        "edge_clearance": 10.0,
    }
    frames = _run(region, (100.0, 50.0), (0.0, -1000.0), settings)
    final = [f for f in frames if f["status"] == "done"][0]
    result = final["result"]
    assert final["alternatives"]
    for alt in final["alternatives"]:
        assert result["peak_load"] <= alt["peak_load"] + 1e-9
        assert alt["n"] == result["n"]

    per_n = {p["n"]: p for p in final["per_n"]}
    assert per_n[3]["seeds"] > 0
    assert abs(per_n[3]["best_peak"] - result["peak_load"]) < 1e-9


def test_per_n_summary_covers_the_whole_sweep():
    region = build_legal_region(RECT, [], edge_clearance=10.0)
    settings = {
        "min_spacing": 15.0, "load_ceiling": 400.0,
        "n_min": 2, "n_max": 6, "seeds_per_n": 15, "iterations": 60,
        "edge_clearance": 10.0,
    }
    frames = _run(region, (100.0, 50.0), (0.0, -1000.0), settings)
    final = [f for f in frames if f["status"] == "done"][0]
    ns = [p["n"] for p in final["per_n"]]
    assert ns == [2, 3, 4, 5, 6]
    # every n keeps live seeds all the way through the mid-run cull
    assert all(p["seeds"] > 0 for p in final["per_n"])
    # N=2 cannot beat |F|/2 = 500 > ceiling; N=3 can (1000/3 = 333)
    assert final["result"]["n"] == 3


def test_stop_flag_still_returns_best_so_far():
    region = build_legal_region(RECT, [], edge_clearance=10.0)
    settings = {
        "min_spacing": 15.0, "load_ceiling": 5000.0,
        "n_min": 2, "n_max": 4, "seeds_per_n": 10, "iterations": 500,
        "edge_clearance": 10.0,
    }
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 25

    async def go():
        return [f async for f in run_optimization(
            region, [_case((100.0, 50.0), (0.0, -1000.0))], settings,
            stream_every=1000, should_stop=should_stop)]

    frames = asyncio.run(go())
    final = [f for f in frames if f["status"] == "done"][0]
    assert final["stopped"] is True
    assert final["result"] is not None
    assert final["result"]["n"] >= 2


def test_timing_default_batch(capsys):
    """Not an assertion of speed so much as a reported number for the default load."""
    region = build_legal_region(RECT, [], edge_clearance=10.0)
    settings = {
        "min_spacing": 15.0, "load_ceiling": 400.0,
        "n_min": 2, "n_max": 8, "seeds_per_n": 143, "iterations": 300,
        "edge_clearance": 10.0,
    }
    t0 = time.perf_counter()
    frames = _run(region, (100.0, 50.0), (0.0, -1000.0), settings)
    elapsed = time.perf_counter() - t0
    final = [f for f in frames if f["status"] == "done"][0]
    with capsys.disabled():
        print(f"\n[timing] B={final['batch_size']} (seeded ~1001), N_max=8, 300 iters, "
              f"CPU: {elapsed:.2f}s -> N={final['result']['n']} "
              f"peak={final['result']['peak_load']:.2f}")
    assert final["result"] is not None


# --------------------------------------------------------------------------
# Directional (ray) distance field + bearing / tear-out model
# --------------------------------------------------------------------------

BEAR_OUTER = [(0, 0), (200, 0), (200, 100), (0, 100)]
BEAR_HOLE = [(90, 40), (110, 40), (110, 60), (90, 60)]


BEAR_RAY_RES = 192
# the ray grid spans the 200-wide bbox plus its margin, so one cell is a touch
# over 1 unit; the field is min-pooled per cell, which makes it *conservative*
# to within about that.
BEAR_CELL = 1.3


def _bearing_fields(n_dirs=32, ray_resolution=BEAR_RAY_RES, ray_cap=None):
    material = Polygon(BEAR_OUTER, [BEAR_HOLE])
    legal = material.buffer(-10)
    return material, build_region_sdf(legal, resolution=256, material_region=material,
                                      n_dirs=n_dirs, ray_resolution=ray_resolution,
                                      ray_cap=ray_cap)


def _assert_conservative(got, truth, tol=BEAR_CELL, what=""):
    """The ray field must never over-report clear distance (that would overstate
    bearing capacity), and must not under-report by more than ~a cell."""
    assert got <= truth + 1e-6, f"{what}: {got} over-reports {truth}"
    assert got >= truth - tol, f"{what}: {got} under-reports {truth} by > {tol}"


def test_ray_distance_known_directions_on_rect_with_hole():
    _, fields = _bearing_fields()
    assert fields.has_ray_field and fields.n_dirs == 32

    P = torch.tensor([[40.0, 50.0]] * 3 + [[60.0, 50.0]], dtype=torch.float64)
    ang = torch.tensor([math.pi, 0.0, math.pi / 2, 0.0], dtype=torch.float64)
    d, kind = fields.ray_distance(P, ang, with_kind=True)

    _assert_conservative(d[0].item(), 40.0, what="west to the left wall")
    _assert_conservative(d[1].item(), 50.0, what="east to the hole wall at x=90")
    _assert_conservative(d[2].item(), 50.0, what="north to the top edge")
    _assert_conservative(d[3].item(), 30.0, what="east from x=60 to the hole")
    # and the limiter classification follows the ring that was hit
    assert kind[0].item() == 0.0 and kind[2].item() == 0.0     # outer profile
    assert kind[1].item() == 1.0 and kind[3].item() == 1.0     # interior hole


def test_ray_distance_between_direction_slices_stays_conservative():
    """K=32 gives one direction every 11.25 deg, and the field is pooled to the
    cell minimum, so an angle landing between two slices reads the shorter of the
    rays around it -- close to the analytic distance, never longer than it."""
    _, fields = _bearing_fields()
    # From (40, 20), 185.625 deg is exactly halfway between the 180 and 191.25
    # slices; both reach the left wall, so the answer is 40 / cos(5.625 deg).
    P = torch.tensor([[40.0, 20.0]], dtype=torch.float64)

    def at(deg):
        return fields.ray_distance(P, torch.tensor([math.radians(deg)],
                                                   dtype=torch.float64)).item()

    lo, hi, mid = at(180.0), at(191.25), at(185.625)
    _assert_conservative(lo, 40.0, what="180 deg")
    _assert_conservative(hi, 40.0 / math.cos(math.radians(11.25)), what="191.25 deg")
    _assert_conservative(mid, 40.0 / math.cos(math.radians(5.625)), what="185.625 deg")


def test_ray_distance_is_differentiable_in_position():
    _, fields = _bearing_fields()
    p = torch.tensor([[40.0, 50.0]], dtype=torch.float64, requires_grad=True)
    fields.ray_distance(p, torch.tensor([0.0], dtype=torch.float64)).sum().backward()
    # heading east, moving east shortens the distance to the hole wall
    assert p.grad[0, 0].item() < 0


def test_adjacent_bolt_clear_distance_is_spacing_minus_diameter():
    """Two bolts in line along the load direction: the leading bolt's tear-out path
    is cut short by the trailing hole, at edge-to-edge distance spacing - d."""
    _, fields = _bearing_fields()
    d = 6.0
    spacing = 25.0
    P = torch.tensor([[[40.0, 50.0], [40.0 + spacing, 50.0]]], dtype=torch.float64)
    M = torch.tensor([[True, True]])
    # hand-made resultants: the *load* on both bolts is due -X, so both bolts
    # *bear* due +X -- see the sign convention test below.
    total = torch.tensor([[[[-100.0, 0.0], [-100.0, 0.0]]]], dtype=torch.float64)
    mags = torch.tensor([[[100.0, 100.0]]], dtype=torch.float64)

    lc, limiter, angles = bearing_clear_distance(P, M, total, mags, fields, d)
    assert abs(angles[0, 0, 0].item()) < 1e-12          # bearing due +X, not -X
    # bolt 0 is blocked by bolt 1 at spacing - d; bolt 1 sees the hole wall at x=90
    assert abs(lc[0, 0, 0].item() - (spacing - d)) < 1e-6
    assert limiter[0, 0, 0].item() == LIMITER_BOLT
    _assert_conservative(lc[0, 0, 1].item(), 90.0 - 65.0 - d / 2,
                         what="bolt 1 east to the hole")
    assert limiter[0, 0, 1].item() == LIMITER_HOLE

    # pull the bolts apart along the same line and the clear distance follows
    P2 = P.clone()
    P2[0, 1, 0] = 40.0 + 2 * spacing
    lc2, _, _ = bearing_clear_distance(P2, M, total, mags, fields, d)
    assert abs(lc2[0, 0, 0].item() - (2 * spacing - d)) < 1e-6

    # offset bolt 1 well outside the +-d corridor and it stops limiting bolt 0
    P3 = P.clone()
    P3[0, 1, 1] = 50.0 + 4 * d
    lc3, lim3, _ = bearing_clear_distance(P3, M, total, mags, fields, d)
    assert lc3[0, 0, 0].item() > spacing - d + 1.0
    assert lim3[0, 0, 0].item() == LIMITER_HOLE       # the hole at x=90 takes over


def test_effective_load_is_raw_over_capacity_fraction():
    _, fields = _bearing_fields()
    d, k_min = 6.0, 0.05
    # one bolt comfortably past the 2d saturation point from the left wall and
    # one at x = 13 (lc = 10 < 2d, so clipped).  The load is due +X so both bolts
    # bear due -X, i.e. toward the left wall.  They are offset in y so that
    # neither sits in the other's tear-out corridor -- this test is about the
    # wall, not about adjacent-bolt blocking.
    P = torch.tensor([[[20.0, 50.0], [13.0, 20.0]]], dtype=torch.float64)
    M = torch.tensor([[True, True]])
    total = torch.tensor([[[[100.0, 0.0], [100.0, 0.0]]]], dtype=torch.float64)
    mags = torch.tensor([[[100.0, 100.0]]], dtype=torch.float64)
    eff, lc, k, lim = effective_loads(P, M, total, mags,
                                      {"fields": fields, "d": d, "k_min": k_min})
    assert k[0, 0, 0].item() == 1.0                       # lc >= 2d -> full capacity
    assert abs(eff[0, 0, 0].item() - 100.0) < 1e-6
    lc1 = lc[0, 0, 1].item()
    _assert_conservative(lc1, 13.0 - d / 2, what="bolt 1 west to the left wall")
    assert abs(k[0, 0, 1].item() - lc1 / (2 * d)) < 1e-9
    assert abs(eff[0, 0, 1].item() - 100.0 / k[0, 0, 1].item()) < 1e-6
    assert eff[0, 0, 1].item() > eff[0, 0, 0].item()


def test_bearing_disabled_leaves_the_raw_objective_untouched():
    """`bearing_enabled=False` (and the no-diameter path) must reproduce the plain
    elastic peak exactly -- this is what keeps the pre-bearing tests meaningful."""
    material = Polygon(RECT)
    region = build_legal_region(RECT, [], edge_clearance=10.0)
    base = {
        "min_spacing": 15.0, "load_ceiling": 5000.0,
        "n_min": 3, "n_max": 3, "seeds_per_n": 10, "iterations": 40,
        "edge_clearance": 10.0,
        "seed": 11,
    }

    async def go(settings, mat):
        return [f async for f in run_optimization(
            region, [_case((100.0, 50.0), (0.0, -1000.0))], settings,
            stream_every=1000, material_region=mat)]

    off = [f for f in asyncio.run(go(dict(base, bearing_enabled=False, bolt_diameter=6.0),
                                     material)) if f["status"] == "done"][0]
    # no material region supplied at all -> the model cannot run, same answer
    none = [f for f in asyncio.run(go(dict(base, bolt_diameter=6.0), None))
            if f["status"] == "done"][0]

    assert off["result"]["bearing"] is False
    assert none["result"]["bearing"] is False
    for r in (off["result"], none["result"]):
        pts = [tuple(p) for p in r["positions"]]
        ref = max(_reference_loads(pts, (100.0, 50.0), (0.0, -1000.0)))
        # the reported peak is the plain elastic peak, with no bearing inflation
        assert abs(r["peak_load"] - ref) < 1e-6
        assert abs(r["raw_peak_load"] - r["peak_load"]) < 1e-12
        assert all("lc" not in b for b in r["per_bolt"])
    # supplying a material region only changes the raster bbox, so the two runs
    # agree to well within the optimizer's own convergence noise
    assert abs(off["result"]["peak_load"] - none["result"]["peak_load"]) < 1.0


# --------------------------------------------------------------------------
# Behavioural: on the real sample plate, bearing moves bolts off the edge they
# push toward.
# --------------------------------------------------------------------------

def _sample_plate(edge_clearance=8.0):
    from app.dxf_loader import load_dxf
    from app.geometry import build_material_region
    path = os.path.join(os.path.dirname(__file__), "..", "samples",
                        "lm3_botplate_solidedge.dxf")
    loops = load_dxf(path)["loops"]
    part = max(range(len(loops)), key=lambda i: Polygon(loops[i]).area)
    others = [l for i, l in enumerate(loops) if i != part]
    material = build_material_region(loops[part], [], other_loops=others)
    legal = build_legal_region(loops[part], [], edge_clearance, other_loops=others)
    return material, legal


def test_bearing_pushes_bolts_off_the_edge_they_load_toward(capsys):
    material, legal = _sample_plate(edge_clearance=8.0)
    minx, miny, maxx, maxy = material.bounds
    d = 6.0
    # one case applied near the middle of the plate, pushing straight down at the
    # lower edge -- with the bearing model off the optimizer is free to park
    # bolts right on the shrunk boundary next to that edge.
    cases = [_case(((minx + maxx) / 2, miny + (maxy - miny) * 0.5), (0.0, -4000.0), "Down")]
    base = {
        "min_spacing": 18.0, "load_ceiling": 100000.0,
        "n_min": 4, "n_max": 4, "seeds_per_n": 250, "iterations": 300,
        "edge_clearance": 8.0,
        "bolt_diameter": d, "seed": 5,
    }

    def run(settings):
        async def go():
            return [f async for f in run_optimization(
                legal, cases, settings, stream_every=10000, material_region=material)]
        t0 = time.perf_counter()
        frames = asyncio.run(go())
        return [f for f in frames if f["status"] == "done"][0], time.perf_counter() - t0

    on, t_on = run(dict(base, bearing_enabled=True, seeds_per_n=250))
    off, t_off = run(dict(base, bearing_enabled=False))

    fields = build_region_sdf(legal, resolution=384, material_region=material,
                              n_dirs=32, ray_resolution=256, ray_cap=3.0 * d)

    def min_lc(positions):
        P = torch.tensor([positions], dtype=torch.float64)
        M = torch.ones(1, len(positions), dtype=torch.bool)
        fp = torch.tensor([[cases[0]["x"], cases[0]["y"]]], dtype=torch.float64)
        fv = torch.tensor([[cases[0]["fx"], cases[0]["fy"]]], dtype=torch.float64)
        total, mags = batched_bolt_loads(P, M, fp, fv)
        lc, _, _ = bearing_clear_distance(P, M, total, mags, fields, d)
        return float(lc.min().item())

    lc_on = min_lc(on["result"]["positions"])
    lc_off = min_lc(off["result"]["positions"])

    with capsys.disabled():
        print(f"\n[bearing] sample plate, N=4, 250 seeds: field build "
              f"{on['field_build_s']:.2f}s, run {t_on:.2f}s (off: {t_off:.2f}s)")
        print(f"[bearing]   on : eff peak {on['result']['peak_load']:.1f} "
              f"(raw {on['result']['raw_peak_load']:.1f}), min lc {lc_on:.2f}")
        print(f"[bearing]   off: peak {off['result']['peak_load']:.1f}, min lc {lc_off:.2f}")

    assert on["result"]["bearing"] is True
    # the ray grid is 256 cells on the long side (128 was too coarse next to this
    # plate's small holes -- a ray could graze between two nodes and the field
    # then reported a long, unsafe clear distance)
    assert on["field_build_s"] < 20.0
    # a bolt sitting on the shrunk boundary and pushing straight at the outer
    # edge would only ever get lc = edge_clearance - d/2 = 5.0
    assert lc_on > 8.0 - d / 2
    # and the bearing run really does keep more clear distance than the raw one
    assert lc_on > lc_off
    # every reported bolt carries the full bearing breakdown
    for row in on["result"]["per_bolt"]:
        assert row["limiter"] in ("outer edge", "hole", "adjacent bolt")
        assert abs(row["effective_load"] - row["load"] / row["k"]) < 1e-6
        assert 0.05 - 1e-9 <= row["k"] <= 1.0 + 1e-9
    assert on["result"]["peak_load"] >= on["result"]["raw_peak_load"] - 1e-9


def test_timing_bearing_full_sweep(capsys):
    """Reported number, not an assertion of speed: the default-size batch (~1000
    seeds across the N sweep) on the real sample plate with bearing enabled."""
    material, legal = _sample_plate(edge_clearance=8.0)
    minx, miny, maxx, maxy = material.bounds
    cases = [_case(((minx + maxx) / 2, (miny + maxy) / 2), (0.0, -4000.0), "Down")]
    settings = {
        "min_spacing": 18.0, "load_ceiling": 800.0,
        "n_min": 2, "n_max": 8, "seeds_per_n": 143, "iterations": 300,
        "edge_clearance": 8.0,
        "bolt_diameter": 6.0, "bearing_enabled": True, "seed": 2,
    }

    async def go():
        return [f async for f in run_optimization(
            legal, cases, settings, stream_every=10000, material_region=material)]

    t0 = time.perf_counter()
    final = [f for f in asyncio.run(go()) if f["status"] == "done"][0]
    elapsed = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[timing] bearing on, sample plate, B={final['batch_size']} "
              f"(seeded ~1001), N=2..8, 300 iters, CPU: {elapsed:.2f}s "
              f"(field build {final['field_build_s']:.2f}s) -> N={final['result']['n']} "
              f"eff peak={final['result']['peak_load']:.2f} "
              f"raw={final['result']['raw_peak_load']:.2f}")
    assert final["result"] is not None
    assert final["result"]["bearing"] is True


# --------------------------------------------------------------------------
# Sign convention: bearing is measured where the BOLT pushes on the PLATE
# --------------------------------------------------------------------------

def test_bearing_direction_is_opposite_the_load():
    """A tension plate pulled to the left tears out at its right-hand end: the
    material that fails is on the side *opposite* the pull, because that is the
    side the bolt shank is driven into.

    So for one bolt sitting close to the LEFT wall:
      * pull the plate LEFT  -> the bolt bears RIGHT, away from that wall, and
        has the whole plate behind it: lc is large and k saturates at 1;
      * pull the plate RIGHT -> the bolt bears LEFT, straight at the near wall,
        and lc collapses to (wall distance - d/2).
    """
    _, fields = _bearing_fields()
    d, k_min = 6.0, 0.05
    wall = 13.0                                   # distance from the left wall
    P = torch.tensor([[[wall, 50.0]]], dtype=torch.float64)
    M = torch.tensor([[True]])
    mags = torch.tensor([[[100.0]]], dtype=torch.float64)

    def at(load_vec):
        total = torch.tensor([[[list(load_vec)]]], dtype=torch.float64)
        eff, lc, k, lim = effective_loads(P, M, total, mags,
                                          {"fields": fields, "d": d, "k_min": k_min})
        ang = bearing_clear_distance(P, M, total, mags, fields, d)[2][0, 0, 0].item()
        return lc[0, 0, 0].item(), k[0, 0, 0].item(), eff[0, 0, 0].item(), ang

    lc_left, k_left, eff_left, ang_left = at((-100.0, 0.0))      # plate pulled LEFT
    lc_right, k_right, eff_right, ang_right = at((100.0, 0.0))   # plate pulled RIGHT

    # the reported bearing angle is the reverse of the load
    assert abs(ang_left) < 1e-12                       # load -X -> bearing +X
    assert abs(abs(ang_right) - math.pi) < 1e-12       # load +X -> bearing -X

    # pulled left: bearing runs east across the plate to the hole at x=90
    assert lc_left > 2 * d
    assert k_left == 1.0
    assert abs(eff_left - 100.0) < 1e-6

    # pulled right: bearing runs west into the near wall, one bolt radius short
    _assert_conservative(lc_right, wall - d / 2, what="bearing west into the wall")
    assert k_right < 1.0
    assert eff_right > eff_left


def test_bearing_direction_sign_on_a_whole_layout():
    """Same convention, but through the elastic model rather than a hand-made
    resultant: a pure-translation bolt group loaded due -X must report every
    bearing angle at 0 (due +X), not at pi."""
    pts = [(60.0, 50.0), (140.0, 50.0)]
    P = torch.tensor([pts], dtype=torch.float64)
    M = torch.ones(1, 2, dtype=torch.bool)
    # force applied on the line through the bolt-group centroid -> no moment,
    # so both resultants are pure -X and both bearing directions are pure +X
    fp = torch.tensor([[100.0, 50.0]], dtype=torch.float64)
    fv = torch.tensor([[-1000.0, 0.0]], dtype=torch.float64)
    total, mags = batched_bolt_loads(P, M, fp, fv)
    _, fields = _bearing_fields()
    _, _, angles = bearing_clear_distance(P, M, total, mags, fields, 6.0)
    assert torch.allclose(angles, torch.zeros_like(angles), atol=1e-9)


def test_adjacent_bolt_corridor_fires_at_the_users_numbers():
    """d = 5, min spacing 10: two bolts in line with the bearing direction leave
    only lc = spacing - d = 5 = 1d between the hole edges, i.e. k = 0.5 -- and the
    adjacent bolt, not the plate edge, must be reported as the limiter."""
    _, fields = _bearing_fields()
    d, spacing = 5.0, 10.0
    # both bolts well away from every wall, in line along X
    P = torch.tensor([[[100.0, 20.0], [100.0 - spacing, 20.0]]], dtype=torch.float64)
    M = torch.tensor([[True, True]])
    # load due +X -> bearing due -X -> bolt 0 bears straight at bolt 1
    total = torch.tensor([[[[100.0, 0.0], [100.0, 0.0]]]], dtype=torch.float64)
    mags = torch.tensor([[[100.0, 100.0]]], dtype=torch.float64)

    eff, lc, k, lim = effective_loads(P, M, total, mags,
                                      {"fields": fields, "d": d, "k_min": 0.05})
    assert abs(lc[0, 0, 0].item() - (spacing - d)) < 1e-9      # lc = 5 = 1d
    assert abs(k[0, 0, 0].item() - 0.5) < 1e-9                 # k = lc / 2d = 0.5
    assert abs(eff[0, 0, 0].item() - 200.0) < 1e-6             # load / k
    assert lim[0, 0, 0].item() == LIMITER_BOLT


# --------------------------------------------------------------------------
# Ray-field accuracy against an exact shapely cast, on the real sample plate
# --------------------------------------------------------------------------

def _exact_ray(boundary, px, py, ang, L):
    from shapely.geometry import LineString
    ux, uy = math.cos(ang), math.sin(ang)
    inter = LineString([(px, py), (px + L * ux, py + L * uy)]).intersection(boundary)
    if inter.is_empty:
        return None
    best = None
    for g in getattr(inter, "geoms", [inter]):
        for c in (g.coords if hasattr(g, "coords") else []):
            dd = math.hypot(c[0] - px, c[1] - py)
            if dd > 1e-9 and (best is None or dd < best):
                best = dd
    return best


def test_ray_field_matches_an_exact_cast_on_the_sample_plate(capsys):
    """The field is a discretization of a *discontinuous* function -- swing a ray a
    fraction of a degree past a small hole and the true answer jumps from
    millimetres to the width of the plate.  Two properties are asserted:

    1. it never over-reports (an over-report overstates bearing capacity, which is
       exactly the failure that made every bolt read k = 1), and
    2. its mean error over the range the model can actually see (everything past
       the cap is saturated anyway) is under one cell.
    """
    import shapely
    d = 5.0
    cap = 3.0 * d
    material, legal = _sample_plate(edge_clearance=3.0)
    fields = build_region_sdf(legal, resolution=384, material_region=material,
                              n_dirs=32, ray_resolution=256, ray_cap=cap)
    K = fields.n_dirs
    minx, miny, maxx, maxy = material.bounds
    L = 4 * max(maxx - minx, maxy - miny)
    ny, nx = fields.ray.shape[1], fields.ray.shape[2]
    cell = max((maxx - minx) / (nx - 1), (maxy - miny) / (ny - 1))

    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < 50:
        xs = rng.uniform(minx, maxx, 400)
        ys = rng.uniform(miny, maxy, 400)
        keep = shapely.contains_xy(legal, xs, ys)
        pts.extend(list(zip(xs[keep], ys[keep])))
    pts = pts[:50]
    angs = list(np.arange(K) * (2 * math.pi / K)) + list(rng.uniform(0, 2 * math.pi, 20))

    bnd = material.boundary
    P, A, E = [], [], []
    for (px, py) in pts:
        for a in angs:
            e = _exact_ray(bnd, px, py, float(a), L)
            if e is None:
                continue
            P.append((px, py))
            A.append(float(a))
            E.append(e)

    got = fields.ray_distance(torch.tensor(P, dtype=torch.float64),
                              torch.tensor(A, dtype=torch.float64)).numpy()
    truth = np.minimum(np.array(E), cap)
    signed = got - truth
    err = np.abs(signed)

    with capsys.disabled():
        print(f"\n[ray] sample plate, {len(err)} casts, cell {cell:.3f}: "
              f"mean |err| {err.mean():.3f} ({err.mean() / cell:.2f} cell), "
              f"max over-report {signed.max():.4f}")

    assert signed.max() <= 1e-6, "ray field over-reports clear distance"
    assert err.mean() < cell, "mean ray-field error exceeds one cell"


def test_ray_field_near_the_left_edge_of_the_sample_plate():
    """Spot check the case the bug report was about: 3 mm inside the left edge of
    the plate, bearing due west, the field must read ~3 -- not the plate width."""
    import shapely
    material, _ = _sample_plate(edge_clearance=3.0)
    fields = build_region_sdf(material, resolution=384, material_region=material,
                              n_dirs=32, ray_resolution=256, ray_cap=15.0)
    minx, miny, maxx, maxy = material.bounds
    checked = 0
    for frac in (0.3, 0.5, 0.7):
        y = miny + frac * (maxy - miny)
        xs = np.linspace(minx, maxx, 4000)
        inside = shapely.contains_xy(material, xs, np.full_like(xs, y))
        if not inside.any():
            continue
        x0 = float(xs[np.argmax(inside)])
        got = float(fields.ray_distance(
            torch.tensor([[x0 + 3.0, y]], dtype=torch.float64),
            torch.tensor([math.pi], dtype=torch.float64))[0])
        assert 1.5 <= got <= 3.1, f"west from 3 mm inside the left edge read {got}"
        checked += 1
    assert checked == 3


# --------------------------------------------------------------------------
# User-placed (fixed) bolts
# --------------------------------------------------------------------------

FIXED_A = (60.0, 30.0)
FIXED_B = (140.0, 70.0)


def _fixed_settings(**over):
    s = {
        "min_spacing": 15.0, "load_ceiling": 400.0,
        "n_min": 4, "n_max": 4, "seeds_per_n": 12, "iterations": 60,
        "edge_clearance": 10.0, "seed": 5,
    }
    s.update(over)
    return s


def _run_fixed(settings=None, fixed=(FIXED_A, FIXED_B), region=None):
    region = region if region is not None else build_legal_region(RECT, [], edge_clearance=10.0)
    settings = settings or _fixed_settings()

    async def go():
        return [f async for f in run_optimization(
            region, [_case((100.0, 50.0), (0.0, -1000.0))], settings,
            stream_every=1000, fixed_bolts=[{"x": x, "y": y} for x, y in fixed])]

    return [f for f in asyncio.run(go()) if f["status"] == "done"][0]


def test_fixed_bolts_appear_verbatim_in_the_result():
    """(a) Two fixed bolts + n=4 -> both placed bolts exactly where the user put
    them, plus two the optimizer chose."""
    result = _run_fixed()["result"]
    assert result["n"] == 4
    pts = [tuple(p) for p in result["positions"]]
    assert len(pts) == 4
    assert pts[0] == FIXED_A
    assert pts[1] == FIXED_B
    assert result["fixed_flags"] == [True, True, False, False]
    assert [b["fixed"] for b in result["per_bolt"]] == [True, True, False, False]
    # the free two are genuinely elsewhere
    for p in pts[2:]:
        assert p not in (FIXED_A, FIXED_B)


def test_fixed_bolt_positions_are_bit_identical_across_the_whole_batch():
    """(b) Pinning must hold for every layout in the batch, not just the winner --
    zeroing the gradient alone would still let Adam's stale momentum drift them.

    `pick_distinct` is neutered so `alternatives` reports every surviving layout
    at the winning bolt count rather than five representatives.
    """
    import app.optimizer as opt

    real_pick = opt.pick_distinct
    opt.pick_distinct = lambda cands, threshold, k=5: list(cands)
    try:
        final = _run_fixed(_fixed_settings(seeds_per_n=40, iterations=120))
    finally:
        opt.pick_distinct = real_pick

    assert final["n_fixed"] == 2
    layouts = [final["result"]] + final["alternatives"]
    assert len(layouts) > 8, "expected the whole surviving batch, not a sample"
    for layout in layouts:
        pts = np.array(layout["positions"])
        # bit-identical, not merely close
        assert pts[0].tolist() == list(FIXED_A)
        assert pts[1].tolist() == list(FIXED_B)
        assert layout["fixed_flags"][:2] == [True, True]
        assert not any(layout["fixed_flags"][2:])


def test_fixed_bolts_are_really_in_the_load_calculation():
    """(c) The reported peak must equal a plain unbatched recomputation over all
    four returned bolts -- if the fixed pair were excluded from the centroid or
    the polar moment the numbers would not line up."""
    result = _run_fixed(_fixed_settings(load_ceiling=1e9))["result"]
    pts = [tuple(p) for p in result["positions"]]
    ref = _reference_loads(pts, (100.0, 50.0), (0.0, -1000.0))
    assert abs(result["peak_load"] - max(ref)) < 1e-6
    assert np.allclose(result["per_bolt_loads"], ref, atol=1e-6)


def test_fixed_bolt_outside_the_region_is_rejected_with_a_clear_error():
    """(d) The endpoint names the offending bolt rather than failing later."""
    from fastapi.testclient import TestClient
    from app.main import app as fastapi_app, STATE

    client = TestClient(fastapi_app)
    STATE["loops"] = [[(0, 0), (200, 0), (200, 100), (0, 100), (0, 0)]]
    STATE["roles"] = {0: "part"}
    STATE["settings"]["edge_clearance"] = 10.0
    STATE["fixed_bolts"] = []

    ok = client.post("/fixed-bolts", json={"bolts": [{"x": 60, "y": 30}]})
    assert ok.status_code == 200
    assert ok.json()["fixed_bolts"] == [{"x": 60.0, "y": 30.0}]

    # 2 units from the wall, well inside the 10-unit clearance
    bad = client.post("/fixed-bolts", json={"bolts": [{"x": 60, "y": 30}, {"x": 2, "y": 50}]})
    assert bad.status_code == 400
    detail = bad.json()["detail"]
    assert "Fixed bolt 2" in detail
    assert "outside the legal placement region" in detail
    # the rejected list was not committed
    assert client.get("/fixed-bolts").json()["fixed_bolts"] == [{"x": 60.0, "y": 30.0}]
    STATE["fixed_bolts"] = []
    STATE["roles"] = {}
    STATE["loops"] = []


def test_seeds_never_place_a_free_bolt_within_min_spacing_of_a_fixed_bolt():
    """(e) The seed pool drops everything inside the exclusion disc around each
    fixed bolt, so no layout starts out violating spacing against a bolt that
    cannot move out of the way."""
    region = build_legal_region(RECT, [], edge_clearance=10.0)
    fixed = np.array([FIXED_A, FIXED_B])
    min_spacing = 18.0
    rng = np.random.default_rng(1)
    seeds, masks, ns = build_seeds(region, 3, 7, 25, min_spacing, rng, fixed=fixed)
    assert len(seeds) > 0
    for b in range(len(seeds)):
        live = seeds[b][masks[b]]
        assert np.allclose(live[:2], fixed)
        free = live[2:]
        if not len(free):
            continue
        d = np.linalg.norm(free[:, None, :] - fixed[None, :, :], axis=2)
        assert d.min() >= min_spacing - 1e-9


# --------------------------------------------------------------------------
# Review fixes: unit independence and degenerate sweep bounds
# --------------------------------------------------------------------------

def test_soft_peak_is_independent_of_load_units():
    """The logsumexp temperature is relative to the layout's own peak, so the
    soft objective scales exactly with the loads and its bias over the hard max
    is the same few percent whether the loads are entered in N or in kN.  (With
    an absolute temperature, loads of order 1 carried a ~60 % bias.)"""
    P = torch.tensor([[(20.0, 30.0), (150.0, 25.0), (120.0, 80.0), (60.0, 60.0)]],
                     dtype=torch.float64)
    M = torch.ones(1, 4, dtype=torch.bool)
    fp = torch.tensor([[60.0, 90.0]], dtype=torch.float64)
    out = {}
    for scale in (1.0, 1000.0):
        fv = torch.tensor([[0.3 * scale, -0.7 * scale]], dtype=torch.float64)
        _, mags = batched_bolt_loads(P, M, fp, fv)
        soft, hard = peak_loads(mags, M)
        out[scale] = (soft.item(), hard.item())
        assert soft.item() >= hard.item() - 1e-12
        assert (soft.item() - hard.item()) / hard.item() < 0.02
    assert abs(out[1000.0][0] / out[1.0][0] - 1000.0) < 1e-9
    assert abs(out[1000.0][1] / out[1.0][1] - 1000.0) < 1e-9


def test_n_min_below_one_is_clamped_not_crashed():
    """n_min = 0 used to seed empty layouts whose peak is -inf; those then won
    every comparison and the run died in the alternatives clustering."""
    region = build_legal_region(RECT, [], edge_clearance=10.0)
    settings = {
        "min_spacing": 15.0, "load_ceiling": 5000.0,
        "n_min": 0, "n_max": 2, "seeds_per_n": 4, "iterations": 10,
        "edge_clearance": 10.0,
    }
    frames = _run(region, (100.0, 50.0), (0.0, -1000.0), settings, stream_every=5)
    final = frames[-1]
    assert final["status"] == "done" and final["result"] is not None
    assert final["result"]["n"] >= 1
    assert [p["n"] for p in final["per_n"]] == [1, 2]
    assert final["n_clamped"] is True
    for f in frames:
        if f["status"] == "progress":
            assert math.isfinite(f["best_peak"])


def test_search_is_invariant_to_drawing_units():
    """The same plate in mm and in inches (with the force point, spacing and
    clearance converted too) must find the same layout: every length the
    optimizer uses -- above all Adam's step -- has to scale with the drawing.
    An absolute 1.0 floor on the learning rate meant a 1 mm step on the metric
    plate became a 1 inch step on the imperial one."""
    k = 1.0 / 25.4
    mm = build_legal_region(RECT, [], edge_clearance=10.0)
    inch = build_legal_region([(x * k, y * k) for x, y in RECT], [], edge_clearance=10.0 * k)
    base = {"load_ceiling": 1e9, "n_min": 4, "n_max": 4, "seeds_per_n": 12,
            "iterations": 80, "seed": 3}
    r_mm = _run(mm, (100.0, 50.0), (0.0, -1000.0),
                dict(base, min_spacing=15.0, edge_clearance=10.0))[-1]["result"]
    r_in = _run(inch, (100.0 * k, 50.0 * k), (0.0, -1000.0),
                dict(base, min_spacing=15.0 * k, edge_clearance=10.0 * k))[-1]["result"]
    assert abs(r_in["peak_load"] / r_mm["peak_load"] - 1.0) < 1e-3
    p_mm = np.sort(np.array(r_mm["positions"]), axis=0)
    p_in = np.sort(np.array(r_in["positions"]), axis=0) / k
    assert np.abs(p_mm - p_in).max() < 0.2   # 0.2 mm on a 200 mm plate
