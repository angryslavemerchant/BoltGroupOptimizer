import math
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ezdxf

from app.dxf_loader import load_dxf

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def test_l_bracket_chains_with_tiny_gaps():
    data = load_dxf(os.path.join(SAMPLES, "l_bracket.dxf"), gap_tol=0.01)
    # the bracket outline (with tiny 0.001 gaps) + circle + keepout rect should all close
    # at least 3 closed loops: bracket outline, circle, keepout rectangle
    assert len(data["loops"]) >= 3
    # the stray line (300,300)-(320,320) must not be closed
    assert len(data["open_chains"]) >= 1


def test_open_loop_reported():
    data = load_dxf(os.path.join(SAMPLES, "plate_open_loop.dxf"), gap_tol=0.01)
    # outer plate rectangle should close
    assert len(data["loops"]) >= 1
    # the inner loop has a 5-unit gap, should NOT be closed -> appears as open chain
    assert len(data["open_chains"]) >= 1
    for chain in data["open_chains"]:
        dx = chain[0][0] - chain[-1][0]
        dy = chain[0][1] - chain[-1][1]
        gap = (dx ** 2 + dy ** 2) ** 0.5
        assert gap > 0.01


def test_snap_points_include_circle_centers():
    data = load_dxf(os.path.join(SAMPLES, "plate_with_holes.dxf"), gap_tol=0.01)
    centers = [p for p in data["snap_points"] if p["kind"] == "center"]
    assert any(abs(p["x"] - 50) < 1e-6 and abs(p["y"] - 50) < 1e-6 for p in centers)
    assert any(abs(p["x"] - 150) < 1e-6 and abs(p["y"] - 50) < 1e-6 for p in centers)


def test_units_reported():
    data = load_dxf(os.path.join(SAMPLES, "plate_with_holes.dxf"))
    assert data["units"] == "mm"


def test_plate_with_holes_loop_count():
    data = load_dxf(os.path.join(SAMPLES, "plate_with_holes.dxf"))
    # outer rect + 2 circles = 3 closed loops
    assert len(data["loops"]) == 3
    assert len(data["open_chains"]) == 0


def test_lm3_botplate_large_file_performance():
    """Real-world fixture: ~4,872 loose LINE entities, no polylines.

    This used to hang the old O(n^2) greedy chainer (>2 min, never returned).
    It must now load in well under the 5s ceiling.
    """
    path = os.path.join(SAMPLES, "lm3_botplate.dxf")
    start = time.perf_counter()
    data = load_dxf(path, gap_tol=0.01)
    elapsed = time.perf_counter() - start

    print(f"\nlm3_botplate: {elapsed:.3f}s, "
          f"loops={len(data['loops'])}, "
          f"open_chains={len(data['open_chains'])}, "
          f"bbox={data['bbox']}")

    assert elapsed < 5.0
    assert len(data["loops"]) >= 1
    assert data["bbox"] is not None
    for key in ("minx", "miny", "maxx", "maxy"):
        assert math.isfinite(data["bbox"][key])
    # $INSUNITS is unset in this drawing; it must be reported as unknown,
    # not silently defaulted to a real unit like mm.
    assert data["units"] == "unknown"


def test_lm3_botplate_solidedge_insert_expansion():
    """Real-world fixture: Solid Edge export whose model space is a single
    INSERT of block 'SE' (13 ARC, 7 CIRCLE, 5 LINE). Before INSERT expansion
    was added, this loaded 0 loops / 0 snap points / no bbox.
    """
    path = os.path.join(SAMPLES, "lm3_botplate_solidedge.dxf")
    data = load_dxf(path, gap_tol=0.01)

    print(f"\nlm3_botplate_solidedge: loops={len(data['loops'])}, "
          f"open_chains={len(data['open_chains'])}, bbox={data['bbox']}, "
          f"units={data['units']}")

    assert len(data["loops"]) >= 1
    centers = [p for p in data["snap_points"] if p["kind"] == "center"]
    assert len(centers) >= 7
    assert data["bbox"] is not None
    for key in ("minx", "miny", "maxx", "maxy"):
        assert math.isfinite(data["bbox"][key])
    # This file has an explicit $INSUNITS (4 -> mm); confirm it's honored.
    assert data["units"] == "mm"


def test_loops_contain_no_duplicate_rings():
    """`polygonize_full` emits a hole ring twice -- as the interior of the face
    around it and as the exterior of the face inside it. Both copies used to reach
    the UI, where an even-odd fill cancelled them back to "filled", so a slot in
    this plate highlighted as solid material. Every ring must appear exactly once.
    """
    from app.dxf_loader import _ring_signature

    data = load_dxf(os.path.join(SAMPLES, "lm3_botplate_solidedge.dxf"), gap_tol=0.01)
    sigs = [_ring_signature(l) for l in data["loops"]]
    assert len(sigs) == len(set(sigs))
    # 7 circles + plate outline + the slot ring
    assert len(data["loops"]) == 9


def test_insert_expansion_transforms_circle_center():
    """Synthetic fixture: a block holding a circle at (10, 0), inserted at
    (100, 100) and rotated 90 degrees. The world-space center snap point
    must reflect the rotation + translation, not the raw block-local center.
    """
    doc = ezdxf.new()
    block = doc.blocks.new(name="ROTBLOCK")
    block.add_circle(center=(10, 0), radius=2)

    msp = doc.modelspace()
    msp.add_blockref("ROTBLOCK", insert=(100, 100), dxfattribs={"rotation": 90})

    tmp_path = os.path.join(SAMPLES, "_tmp_insert_rotation_test.dxf")
    doc.saveas(tmp_path)
    try:
        data = load_dxf(tmp_path)
        centers = [p for p in data["snap_points"] if p["kind"] == "center"]
        assert any(
            abs(p["x"] - 100) < 1e-6 and abs(p["y"] - 110) < 1e-6
            for p in centers
        ), centers
    finally:
        os.remove(tmp_path)


def test_loader_reports_ignored_entities_and_paper_space():
    """A drawing whose only model-space content is unsupported (TEXT) and whose
    geometry sits in a paper-space layout must say so rather than load as an
    unexplained empty drawing."""
    doc = ezdxf.new("R2010")
    doc.modelspace().add_text("just a note")
    layout = doc.layouts.new("SHEET1")
    layout.add_circle((10, 10), 5)
    tmp_path = os.path.join(SAMPLES, "_tmp_ignored_test.dxf")
    doc.saveas(tmp_path)
    try:
        data = load_dxf(tmp_path)
        assert data["loops"] == []
        assert data["ignored"] == {"TEXT": 1}
        assert data["paperspace_entities"] >= 1
        assert "TEXT" in data["message"] and "paper space" in data["message"]
    finally:
        os.remove(tmp_path)


def test_ring_drawn_twice_from_different_sources_is_reported_once():
    """The same rectangle as a closed LWPOLYLINE and as four loose LINEs: the
    chained copy is snapped onto the gap-tolerance grid while the polyline keeps
    its raw coordinates, so the two only match when deduped at that grid."""
    pts = [(0.123, 0.456), (200.123, 0.456), (200.123, 100.456), (0.123, 100.456)]
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_lwpolyline(pts, close=True)
    for i in range(4):
        msp.add_line(pts[i], pts[(i + 1) % 4])
    tmp_path = os.path.join(SAMPLES, "_tmp_dup_ring_test.dxf")
    doc.saveas(tmp_path)
    try:
        data = load_dxf(tmp_path, gap_tol=0.01)
        assert len(data["loops"]) == 1
        assert data["open_chains"] == []
    finally:
        os.remove(tmp_path)
