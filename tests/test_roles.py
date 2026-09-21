"""Loop role semantics: hole (default) vs keepout (solid, placement-only) vs
ignore (no geometric effect at all), on the real sample plate with its kidney
slot, plus the per-(case, bolt) bearing direction the canvas ticks need.
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app as fastapi_app, STATE

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "lm3_botplate_solidedge.dxf")


@pytest.fixture()
def client():
    c = TestClient(fastapi_app)
    with open(SAMPLE, "rb") as fh:
        res = c.post("/load-dxf", files={"file": ("plate.dxf", fh.read())})
    assert res.status_code == 200
    loops = res.json()["loops"]
    # the part is the biggest-area loop; the kidney slot is the next biggest
    # (674.8 area / 129 verts vs the round holes' 65-vert circles)
    from shapely.geometry import Polygon
    by_area = sorted(range(len(loops)), key=lambda i: -Polygon(loops[i]).area)
    part_idx, slot_idx = by_area[0], by_area[1]
    assert c.post("/select", json={"loop_index": part_idx, "role": "part"}).status_code == 200
    assert c.post("/datum", json={"x": 0.0, "y": 0.0}).status_code == 200
    assert c.post("/settings", json={"edge_clearance": 8.0, "min_spacing": 15.0,
                                     "bolt_diameter": 6.0}).status_code == 200
    yield c, part_idx, slot_idx
    STATE["layout"] = None
    STATE["last_result"] = None
    STATE["roles"] = {}
    STATE["loops"] = []
    STATE["forces"] = []
    STATE["fixed_bolts"] = []
    STATE["_cache"] = {}


def _region_area(c):
    data = c.get("/region").json()
    if data["empty"]:
        return 0.0
    from shapely.geometry import Polygon as P
    total = 0.0
    for poly in data["polygons"]:
        total += P(poly["exterior"], poly["holes"]).area
    return total


def test_unmarked_slot_is_a_hole_in_both_regions(client):
    """Default (unmarked): both the legal region and the material region treat
    the slot as removed material, same as before this change."""
    c, part_idx, slot_idx = client
    area_with_slot_as_hole = _region_area(c)
    # now mark it ignore -- the legal region should grow, since the slot area
    # is no longer subtracted
    assert c.post("/select", json={"loop_index": slot_idx, "role": "ignore"}).status_code == 200
    area_ignored = _region_area(c)
    assert area_ignored > area_with_slot_as_hole


def test_keepout_slot_excluded_from_legal_but_present_in_material(client):
    c, part_idx, slot_idx = client

    # legal region must not intersect the slot loop
    from shapely.geometry import Polygon as P
    slot_poly = P(STATE["loops"][slot_idx])
    minx, miny, maxx, maxy = slot_poly.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

    # a bolt just outside the slot, pushed straight at its centroid: with the
    # slot as a default hole the bearing ray hits its near wall almost at once
    # (a short lc); marked keepout the slot is solid, so the same ray must
    # travel through it to the real material edge beyond -- a materially
    # longer lc, and the limiter is no longer "the slot" (whichever edge it
    # names, it's a farther one).
    bx, by = minx - 12.0, cy
    # bearing direction is -load, so a force pulling the bolt in -x makes it
    # push back in +x -- toward the slot, which sits to the right of this point
    assert c.post("/forces", json={"forces": [
        {"name": "Push", "x": bx, "y": by, "fx": -1000.0, "fy": 0.0}]}).status_code == 200

    default_ev = c.post("/evaluate", json={"bolts": [{"x": bx, "y": by, "fixed": False}]}).json()
    lc_default = default_ev["per_bolt"][0]["lc"]

    assert c.post("/select", json={"loop_index": slot_idx, "role": "keepout"}).status_code == 200

    region_data = c.get("/region").json()
    for poly in region_data["polygons"]:
        assert not P(poly["exterior"], poly["holes"]).intersects(slot_poly.buffer(-1e-6))

    # material region (bearing edge) must *include* the slot as solid material,
    # i.e. the slot area is not punched out of it
    from app.geometry import build_material_region
    part_loop = STATE["loops"][part_idx]
    others = [l for i, l in enumerate(STATE["loops"]) if i not in (part_idx, slot_idx)]
    material_without_slot_hole = build_material_region(part_loop, [], other_loops=others)
    assert material_without_slot_hole.intersects(slot_poly.buffer(-1e-6))

    keepout_ev = c.post("/evaluate", json={"bolts": [{"x": bx, "y": by, "fixed": False}]}).json()
    lc_keepout = keepout_ev["per_bolt"][0]["lc"]
    assert keepout_ev["bolts"][0]["in_region"] is True
    assert lc_keepout > lc_default + 1.0


def test_ignore_slot_drops_from_hole_count_and_affects_neither_region(client):
    c, part_idx, slot_idx = client
    default_area = _region_area(c)
    assert c.post("/select", json={"loop_index": slot_idx, "role": "ignore"}).status_code == 200
    ignored_area = _region_area(c)
    assert ignored_area > default_area

    from app.geometry import build_material_region, build_legal_region
    part_loop = STATE["loops"][part_idx]
    others_excluding_slot = [l for i, l in enumerate(STATE["loops"])
                             if i not in (part_idx, slot_idx)]
    material_no_slot = build_material_region(part_loop, [], other_loops=others_excluding_slot)
    legal_no_slot = build_legal_region(part_loop, [], 8.0, other_loops=others_excluding_slot)
    from shapely.geometry import Polygon as P
    assert material_no_slot.area == pytest.approx(
        build_material_region(part_loop, [], other_loops=others_excluding_slot).area)
    assert legal_no_slot.area == pytest.approx(ignored_area, rel=1e-6)


def test_default_role_still_excludes_the_slot_from_both_regions(client):
    """Existing behaviour, unchanged: an unmarked interior loop is a hole in
    both the legal region and the material region."""
    c, part_idx, slot_idx = client
    from app.geometry import build_material_region
    from shapely.geometry import Polygon as P
    part_loop = STATE["loops"][part_idx]
    others = [l for i, l in enumerate(STATE["loops"]) if i != part_idx]
    material = build_material_region(part_loop, [], other_loops=others)
    slot_poly = P(STATE["loops"][slot_idx])
    assert not material.intersects(slot_poly.buffer(-1e-6))


def test_per_case_bolt_dirs_present_in_evaluate_payload(client):
    """Task 2: the payload must carry a per-(case, bolt) bearing direction, not
    just the governing case's, so the canvas can draw one tick per case."""
    c, part_idx, slot_idx = client
    assert c.post("/forces", json={"forces": [
        {"name": "A", "x": 10.0, "y": 10.0, "fx": 1000.0, "fy": 0.0},
        {"name": "B", "x": 10.0, "y": 20.0, "fx": 0.0, "fy": 1000.0},
    ]}).status_code == 200
    bolts = [{"x": 30.0, "y": 30.0, "fixed": False}, {"x": 60.0, "y": 30.0, "fixed": False}]
    data = c.post("/evaluate", json={"bolts": bolts}).json()
    assert len(data["per_case"]) == 2
    for pc in data["per_case"]:
        assert "bolt_dirs" in pc
        assert len(pc["bolt_dirs"]) == len(bolts)
        for v in pc["bolt_dirs"]:
            assert len(v) == 2
            mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
            assert mag == pytest.approx(1.0, abs=1e-6)
