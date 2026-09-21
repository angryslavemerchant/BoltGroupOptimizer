"""POST /evaluate, the layout state model, and the export guard.

These cover the freely-draggable-bolt workflow: the frontend posts the layout it
is currently showing on every drag frame and must get back exactly the numbers
the optimizer itself reported for that layout -- so the two share one scoring
path -- fast enough to run at ~30 Hz.
"""
import os
import sys
import time

import ezdxf
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app as fastapi_app, STATE

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "plate_with_holes.dxf")

# The 200x100 sample plate has two 16-unit holes centred at (50, 50) and (150, 50).
HOLE = (50.0, 50.0)


@pytest.fixture()
def client():
    c = TestClient(fastapi_app)
    with open(SAMPLE, "rb") as fh:
        res = c.post("/load-dxf", files={"file": ("plate_with_holes.dxf", fh.read())})
    assert res.status_code == 200
    # loop 0 is the outer 200x100 rectangle (see the loop dump in the sample)
    assert c.post("/select", json={"loop_index": 0, "role": "part"}).status_code == 200
    assert c.post("/datum", json={"x": 0.0, "y": 0.0}).status_code == 200
    assert c.post("/forces", json={"forces": [
        {"name": "Extended", "x": 200.0, "y": 50.0, "fx": 0.0, "fy": -1000.0},
        {"name": "Closed", "x": 200.0, "y": 50.0, "fx": 700.0, "fy": -700.0},
    ]}).status_code == 200
    assert c.post("/settings", json={
        "edge_clearance": 8.0, "min_spacing": 20.0, "bolt_diameter": 6.0,
        "bearing_enabled": True, "load_ceiling": 700.0,
        "n_min": 5, "n_max": 5, "seeds_per_n": 8, "iterations": 40,
    }).status_code == 200
    yield c
    STATE["layout"] = None
    STATE["last_result"] = None
    STATE["roles"] = {}
    STATE["loops"] = []
    STATE["forces"] = []
    STATE["fixed_bolts"] = []
    STATE["_cache"] = {}


def _run(client):
    """Drive one optimization over the websocket and return the final frame."""
    with client.websocket_connect("/optimize") as ws:
        while True:
            frame = ws.receive_json()
            assert frame["status"] != "error", frame.get("message")
            if frame["status"] == "done":
                return frame


def _as_bolts(result):
    flags = result.get("fixed_flags") or [False] * len(result["positions"])
    return [{"x": p[0], "y": p[1], "fixed": bool(flags[i])}
            for i, p in enumerate(result["positions"])]


def test_evaluate_reproduces_the_optimizers_own_numbers(client):
    """(a) Same layout in, same numbers out -- to 1e-9, because it is literally the
    same scoring code, not a re-implementation."""
    final = _run(client)
    result = final["result"]
    assert result is not None

    ev = client.post("/evaluate", json={"bolts": _as_bolts(result)})
    assert ev.status_code == 200, ev.text
    data = ev.json()

    assert data["peak_eff"] == pytest.approx(result["peak_load"], abs=1e-9)
    assert data["peak_raw"] == pytest.approx(result["raw_peak_load"], abs=1e-9)
    assert len(data["per_bolt"]) == len(result["per_bolt"])
    for got, want in zip(data["per_bolt"], result["per_bolt"]):
        assert got["effective_load"] == pytest.approx(want["effective_load"], abs=1e-9)
        assert got["load"] == pytest.approx(want["load"], abs=1e-9)
        assert got["lc"] == pytest.approx(want["lc"], abs=1e-9)
        assert got["k"] == pytest.approx(want["k"], abs=1e-9)
        assert got["limiter"] == want["limiter"]
    assert len(data["per_case"]) == len(result["per_case"])
    for got, want in zip(data["per_case"], result["per_case"]):
        assert got["name"] == want["name"]
        assert got["peak"] == pytest.approx(want["peak"], abs=1e-9)
        assert got["raw_peak"] == pytest.approx(want["raw_peak"], abs=1e-9)
    assert data["governing_case"] == result["governing_case"]
    assert data["feasible"] is True
    assert all(b["in_region"] for b in data["bolts"])


def test_bolt_dragged_into_a_hole_is_infeasible_and_blocks_export(client):
    """(b) Dropping a bolt in a keepout/hole is allowed (no snap-back) but is
    reported as illegal, and the DXF export then refuses."""
    final = _run(client)
    result = final["result"]

    # export is fine for the optimizer's own layout
    assert client.get("/export.dxf").status_code == 200

    bolts = _as_bolts(result)
    bolts[0] = {"x": HOLE[0], "y": HOLE[1], "fixed": False}   # dead centre of a hole

    ev = client.post("/evaluate", json={"bolts": bolts}).json()
    assert ev["bolts"][0]["in_region"] is False
    assert ev["per_bolt"][0]["in_region"] is False
    assert ev["feasible"] is False
    # only the moved bolt is flagged
    assert all(b["in_region"] for b in ev["bolts"][1:])

    lay = client.post("/layout", json={"bolts": bolts, "source": "manual"})
    assert lay.status_code == 200
    assert lay.json()["source"] == "manual"
    assert lay.json()["feasible"] is False

    bad = client.get("/export.dxf")
    assert bad.status_code == 400
    assert "outside the legal placement region" in bad.json()["detail"]


def test_bolts_closer_than_min_spacing_are_flagged(client):
    """Spacing is checked on the edited layout too, not just at optimization time."""
    final = _run(client)
    bolts = _as_bolts(final["result"])
    # park bolt 1 five units from bolt 0 -- well inside the 20-unit minimum
    bolts[1] = {"x": bolts[0]["x"] + 5.0, "y": bolts[0]["y"], "fixed": False}
    ev = client.post("/evaluate", json={"bolts": bolts}).json()
    assert ev["per_bolt"][0]["spacing_ok"] is False
    assert ev["per_bolt"][1]["spacing_ok"] is False
    assert ev["feasible"] is False


def test_evaluate_is_fast_enough_to_drag_at_30hz(client):
    """(c) Under 50 ms per call on CPU for 8 bolts x 2 cases with bearing on, once
    the cached RegionFields is warm (the first call pays for the rasterization)."""
    bolts = [{"x": x, "y": y, "fixed": False}
             for x in (20.0, 80.0, 120.0, 180.0) for y in (20.0, 80.0)]
    assert len(bolts) == 8
    warm = client.post("/evaluate", json={"bolts": bolts})
    assert warm.status_code == 200
    assert warm.json()["bearing"] is True

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        res = client.post("/evaluate", json={"bolts": bolts})
        times.append((time.perf_counter() - t0) * 1000.0)
        assert res.status_code == 200
    best = min(times)
    print(f"\n[evaluate] 8 bolts x 2 cases, bearing on: "
          f"min {best:.1f} ms, median {sorted(times)[len(times)//2]:.1f} ms")
    assert best < 50.0


def test_revert_restores_the_exact_optimizer_layout(client):
    """(d) Revert is exact: the same coordinates, to the bit."""
    final = _run(client)
    result = final["result"]
    before = client.get("/result").json()
    assert before["source"] == "optimized"
    assert before["can_revert"] is False

    bolts = _as_bolts(result)
    bolts[0] = {"x": bolts[0]["x"] + 7.5, "y": bolts[0]["y"] - 3.25, "fixed": False}
    manual = client.post("/layout", json={"bolts": bolts, "source": "manual"}).json()
    assert manual["source"] == "manual"
    assert manual["can_revert"] is True
    assert manual["positions"][0] != result["positions"][0]

    # a second edit keeps the *original* ancestor rather than chaining
    bolts[1] = {"x": bolts[1]["x"] + 1.0, "y": bolts[1]["y"], "fixed": False}
    client.post("/layout", json={"bolts": bolts, "source": "manual"})

    back = client.post("/layout/revert").json()
    assert back["source"] == "optimized"
    assert back["can_revert"] is False
    assert back["positions"] == result["positions"]
    assert client.get("/result").json()["positions"] == result["positions"]
    assert client.get("/export.dxf").status_code == 200


def test_region_fields_are_cached_across_evaluate_calls(client):
    """The drag loop must never rebuild the rasterized fields."""
    bolts = [{"x": 30.0, "y": 30.0, "fixed": False}, {"x": 170.0, "y": 70.0, "fixed": False}]
    client.post("/evaluate", json={"bolts": bolts})
    fields = STATE["_cache"]["fields"]
    client.post("/evaluate", json={"bolts": bolts})
    assert STATE["_cache"]["fields"] is fields
    # ...but a changed edge clearance does invalidate it
    client.post("/settings", json={"edge_clearance": 12.0})
    client.post("/evaluate", json={"bolts": bolts})
    assert STATE["_cache"]["fields"] is not fields


# --------------------------------------------------------------------------
# Review fixes: results follow the current inputs; export integrity
# --------------------------------------------------------------------------

def test_result_and_export_follow_settings_and_force_edits(client):
    """A result is a set of positions; the numbers on it must be recomputed for
    whatever the sidebar says now.  Doubling every force doubles the reported
    peak; raising the clearance under a bolt makes the same layout infeasible
    and blocks the export -- no stale breakdown survives a settings edit."""
    final = _run(client)
    before = client.get("/result").json()
    assert before["peak_eff"] == pytest.approx(final["result"]["peak_load"], abs=1e-9)

    forces = client.get("/forces").json()["forces"]
    for f in forces:
        f["fx"] *= 2.0
        f["fy"] *= 2.0
    assert client.post("/forces", json={"forces": forces}).status_code == 200
    after = client.get("/result").json()
    assert after["positions"] == before["positions"]
    assert after["peak_eff"] == pytest.approx(2.0 * before["peak_eff"], rel=1e-9)
    assert after["peak_raw"] == pytest.approx(2.0 * before["peak_raw"], rel=1e-9)

    # a much larger clearance strands the bolts the optimizer parked near an edge
    assert client.post("/settings", json={"edge_clearance": 25.0}).status_code == 200
    res = client.get("/result")
    assert res.status_code == 200, res.text
    stale = res.json()
    assert stale["feasible"] is False
    assert any(not b["in_region"] for b in stale["bolts"])
    bad = client.get("/export.dxf")
    assert bad.status_code == 400
    assert "outside the legal placement region" in bad.json()["detail"]

    # and putting it back restores the original verdict
    assert client.post("/settings", json={"edge_clearance": 8.0}).status_code == 200
    assert client.get("/result").json()["feasible"] is True


def test_fixed_bolt_on_an_existing_hole_centre_is_legal(client):
    """The main use of a fixed bolt is an existing hole.  A hole with a bolt in
    it is occupied, so it must stop counting as removed material: the bolt is
    accepted, scores as inside the region, and does not bear against its own
    hole wall.  An empty hole stays a hole."""
    res = client.post("/fixed-bolts", json={"bolts": [{"x": HOLE[0], "y": HOLE[1]}]})
    assert res.status_code == 200, res.text
    bolts = [{"x": HOLE[0], "y": HOLE[1], "fixed": True}, {"x": 170.0, "y": 70.0, "fixed": False}]
    data = client.post("/evaluate", json={"bolts": bolts}).json()
    assert data["bolts"][0]["in_region"] is True
    assert data["per_bolt"][0]["limiter"] != "hole"
    assert data["feasible"] is True
    # the other, empty hole at (150, 50) is still removed material
    other = client.post("/evaluate", json={"bolts": [
        {"x": HOLE[0], "y": HOLE[1], "fixed": True}, {"x": 150.0, "y": 50.0, "fixed": False}]}).json()
    assert other["bolts"][1]["in_region"] is False
    # and a bolt away from any hole is validated as before
    bad = client.post("/fixed-bolts", json={"bolts": [{"x": HOLE[0], "y": HOLE[1]}, {"x": 2.0, "y": 50.0}]})
    assert bad.status_code == 400
    STATE["fixed_bolts"] = []


def test_run_reuses_the_cached_fields(client):
    """The websocket run builds the rasterized fields through the same cache the
    drag-evaluate loop reads, so the first drag after a run is instant."""
    _run(client)
    fields = STATE["_cache"].get("fields")
    assert fields is not None
    bolts = [{"x": 30.0, "y": 30.0, "fixed": False}, {"x": 170.0, "y": 70.0, "fixed": False}]
    client.post("/evaluate", json={"bolts": bolts})
    assert STATE["_cache"]["fields"] is fields


def test_evaluate_without_a_datum_reports_absolute_coordinates(client):
    STATE["datum"] = None
    bolts = [{"x": 30.0, "y": 30.0, "fixed": True}, {"x": 170.0, "y": 70.0, "fixed": False}]
    res = client.post("/evaluate", json={"bolts": bolts})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["datum"] is None
    assert data["bolts"][0]["x"] == 30.0 and data["bolts"][0]["y"] == 30.0
    assert data["bolts"][0]["fixed"] is True and data["bolts"][1]["fixed"] is False


def _export_doc(client):
    res = client.get("/export.dxf")
    assert res.status_code == 200, res.text
    path = os.path.join(os.path.dirname(__file__), "_tmp_export_check.dxf")
    with open(path, "wb") as fh:
        fh.write(res.content)
    try:
        return ezdxf.readfile(path)
    finally:
        os.remove(path)


def test_export_contains_every_bolt_case_and_the_datum_axes(client):
    final = _run(client)
    bolts = _as_bolts(final["result"])
    bolts[0]["fixed"] = True
    assert client.post("/layout", json={"bolts": bolts, "source": "manual"}).status_code == 200
    doc = _export_doc(client)
    assert doc.dxfversion >= "AC1015"
    msp = doc.modelspace()
    for layer in ("BOLTS", "FIXED_BOLTS", "DATUM", "FORCE"):
        assert layer in doc.layers
    circles = [e for e in msp.query("CIRCLE") if e.dxf.layer in ("BOLTS", "FIXED_BOLTS")]
    assert len(circles) == len(bolts)
    assert sum(1 for c in circles if c.dxf.layer == "FIXED_BOLTS") == 1
    force_lines = [e for e in msp.query("LINE") if e.dxf.layer == "FORCE"]
    force_text = [e.dxf.text for e in msp.query("TEXT") if e.dxf.layer == "FORCE"]
    assert len(force_lines) == 2 and sorted(force_text) == ["Closed", "Extended"]
    datum_text = [e.dxf.text for e in msp.query("TEXT") if e.dxf.layer == "DATUM"]
    assert {"+X", "+Y", "DATUM"} <= set(datum_text)
    notes = list(msp.query("MTEXT"))
    assert len(notes) == 1
    text = notes[0].text
    assert "UNITS: mm" in text and "Extended" in text and "Closed" in text
    for i in range(len(bolts)):
        assert f"BOLT {i + 1}" in text
    assert "BOLT 1 (fixed)" in text
    assert "LIMITED BY" in text


def test_export_from_an_r12_source_falls_back_to_text_entities():
    """R12 has no MTEXT (or LWPOLYLINE); exporting on top of an R12 drawing used
    to raise DXFVersionError.  The note block becomes stacked TEXT instead."""
    doc = ezdxf.new("R12")
    msp = doc.modelspace()
    pts = [(0, 0), (200, 0), (200, 100), (0, 100)]
    for i in range(4):
        msp.add_line(pts[i], pts[(i + 1) % 4])
    msp.add_circle((50, 50), 8)
    path = os.path.join(os.path.dirname(__file__), "_tmp_r12_source.dxf")
    doc.saveas(path)
    c = TestClient(fastapi_app)
    try:
        with open(path, "rb") as fh:
            res = c.post("/load-dxf", files={"file": ("r12.dxf", fh.read())})
        assert res.status_code == 200
        loops = res.json()["loops"]
        # the part is the loop with the largest extent
        part = max(range(len(loops)),
                   key=lambda i: max(p[0] for p in loops[i]) - min(p[0] for p in loops[i]))
        assert c.post("/select", json={"loop_index": part, "role": "part"}).status_code == 200
        assert c.post("/datum", json={"x": 0.0, "y": 0.0}).status_code == 200
        assert c.post("/forces", json={"forces": [
            {"name": "Pull", "x": 100.0, "y": 50.0, "fx": 0.0, "fy": -1000.0}]}).status_code == 200
        assert c.post("/settings", json={"edge_clearance": 8.0, "min_spacing": 20.0,
                                         "bolt_diameter": 6.0}).status_code == 200
        bolts = [{"x": 20.0, "y": 20.0, "fixed": False}, {"x": 180.0, "y": 80.0, "fixed": False}]
        assert c.post("/layout", json={"bolts": bolts, "source": "manual"}).status_code == 200
        out = _export_doc(c)
        assert out.dxfversion == "AC1009"
        msp = out.modelspace()
        assert len(list(msp.query("MTEXT"))) == 0
        texts = [e.dxf.text for e in msp.query("TEXT") if e.dxf.layer == "DATUM"]
        assert any(t.startswith("BOLT 2") for t in texts)
        assert len([e for e in msp.query("CIRCLE") if e.dxf.layer == "BOLTS"]) == 2
    finally:
        os.remove(path)
        STATE["layout"] = None
        STATE["last_result"] = None
        STATE["roles"] = {}
        STATE["loops"] = []
        STATE["forces"] = []
        STATE["fixed_bolts"] = []
        STATE["_cache"] = {}
