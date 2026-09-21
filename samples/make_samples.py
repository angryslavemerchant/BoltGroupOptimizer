"""Generate test-fixture DXFs for the bolt group optimizer."""
import os
import ezdxf

OUT_DIR = os.path.dirname(__file__)


def make_plate_with_holes():
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4  # mm
    msp = doc.modelspace()
    # rectangular plate 200x100
    pts = [(0, 0), (200, 0), (200, 100), (0, 100), (0, 0)]
    msp.add_lwpolyline(pts, close=True)
    msp.add_circle((50, 50), 8)
    msp.add_circle((150, 50), 8)
    doc.saveas(os.path.join(OUT_DIR, "plate_with_holes.dxf"))


def make_l_bracket():
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4  # mm
    msp = doc.modelspace()
    # L-shape outline as loose LINE entities with tiny 0.001 gaps at some joints
    pts = [(0, 0), (150, 0), (150, 60), (60, 60), (60, 150), (0, 150), (0, 0)]
    gap = 0.001
    n = len(pts) - 1
    for i in range(n):
        p1 = pts[i]
        p2 = pts[i + 1]
        # introduce a tiny gap at a couple of joints by shrinking the segment start
        if i in (1, 3):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = (dx ** 2 + dy ** 2) ** 0.5
            if length > 0:
                ux, uy = dx / length, dy / length
                p1 = (p1[0] + ux * gap, p1[1] + uy * gap)
        msp.add_line(p1, p2)

    msp.add_circle((30, 30), 6)

    # separate small closed rectangle to use as keepout, inside the bracket
    kp = [(20, 100), (40, 100), (40, 120), (20, 120), (20, 100)]
    msp.add_lwpolyline(kp, close=True)

    # stray unconnected line
    msp.add_line((300, 300), (320, 320))

    doc.saveas(os.path.join(OUT_DIR, "l_bracket.dxf"))


def make_plate_with_open_loop():
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4  # mm
    msp = doc.modelspace()
    pts = [(0, 0), (250, 0), (250, 150), (0, 150), (0, 0)]
    msp.add_lwpolyline(pts, close=True)

    # interior loop with an intentional 5-unit gap (not closable)
    inner = [(50, 50), (150, 50), (150, 100), (55, 100)]  # missing final segment back to (50,50) by 5 units
    for i in range(len(inner) - 1):
        msp.add_line(inner[i], inner[i + 1])
    # gap of 5 units between inner[-1] and inner[0]
    last = inner[-1]
    first = inner[0]
    # leave gap: draw a segment that stops 5 units short of first
    dx = first[0] - last[0]
    dy = first[1] - last[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length > 5:
        ux, uy = dx / length, dy / length
        stop = (last[0] + ux * (length - 5), last[1] + uy * (length - 5))
        msp.add_line(last, stop)

    doc.saveas(os.path.join(OUT_DIR, "plate_open_loop.dxf"))


if __name__ == "__main__":
    make_plate_with_holes()
    make_l_bracket()
    make_plate_with_open_loop()
    print("Wrote samples to", OUT_DIR)
