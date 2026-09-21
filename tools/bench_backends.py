"""Time the same optimizer run on several backends, from one identical seed.

    .venv-dml\\Scripts\\python.exe tools\\bench_backends.py --backends cpu directml --seeds 200 1000 4000

Drives `run_optimization` directly (no server), on the real sample plate and
the real load cases, so the reported N / peak / min lc are the numbers the UI
would show. `--seed` is fixed across backends, so any difference in the result
is the backend's, not the seeding's.
"""
import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.backend import available_backends  # noqa: E402
from app.dxf_loader import load_dxf  # noqa: E402
from app.geometry import build_legal_region, build_material_region  # noqa: E402
from app.optimizer import run_optimization  # noqa: E402

DXF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "samples", "lm3_botplate_solidedge.dxf")

CASES = [
    {"name": "Case 1", "x": 278.17, "y": 149.54, "fx": 5000.0 * -1.0, "fy": 0.0},
    {"name": "Case 2", "x": 243.76, "y": 167.45,
     "fx": 5000.0 * -0.573576436, "fy": 5000.0 * 0.819152044},
]

SETTINGS = {
    "edge_clearance": 3.0,
    "min_spacing": 10.0,
    "bolt_diameter": 5.0,
    "bearing_enabled": True,
    "load_ceiling": 1000.0,
    "n_min": 4,
    "n_max": 8,
    "iterations": 500,
    "seed": 12345,
}


def build_region():
    loaded = load_dxf(DXF)
    loops = loaded["loops"]
    # the part outline is the largest closed loop; everything else is a hole
    import shapely
    areas = [abs(shapely.Polygon(l).area) if len(l) >= 3 else 0.0 for l in loops]
    part_idx = max(range(len(loops)), key=lambda i: areas[i])
    part = loops[part_idx]
    holes = [l for i, l in enumerate(loops) if i != part_idx and len(l) >= 3]
    region = build_legal_region(part, [], SETTINGS["edge_clearance"], other_loops=holes)
    material = build_material_region(part, [], other_loops=holes)
    return region, material


def run(region, material, backend, seeds):
    st = dict(SETTINGS, backend=backend, seeds_per_n=seeds)

    async def go():
        out = None
        async for f in run_optimization(region, CASES, st, stream_every=10 ** 9,
                                        material_region=material):
            if f.get("status") in ("done", "error"):
                out = f
        return out

    t0 = time.perf_counter()
    frame = asyncio.run(go())
    wall = time.perf_counter() - t0
    return frame, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[200, 1000, 4000])
    args = ap.parse_args()
    backends = args.backends or available_backends()

    print("available:", available_backends())
    region, material = build_region()
    print(f"region area {region.area:.0f}, material area {material.area:.0f}\n")

    hdr = f"{'seeds':>6} {'backend':<10} {'N':>3} {'eff peak':>10} {'min lc':>8} " \
          f"{'fields s':>9} {'wall s':>8} {'batch':>7}"
    print(hdr)
    print("-" * len(hdr))
    for seeds in args.seeds:
        for b in backends:
            frame, wall = run(region, material, b, seeds)
            if not frame or frame.get("status") == "error" or not frame.get("result"):
                print(f"{seeds:>6} {b:<10} ERROR {frame and frame.get('message')}")
                continue
            r = frame["result"]
            lcs = [row.get("lc") for row in r["per_bolt"] if row.get("lc") is not None]
            print(f"{seeds:>6} {frame['backend']:<10} {r['n']:>3} {r['peak_load']:>10.2f} "
                  f"{(min(lcs) if lcs else float('nan')):>8.2f} "
                  f"{frame['field_build_s']:>9.2f} {wall:>8.2f} {frame['batch_size']:>7}")
        print()


if __name__ == "__main__":
    main()
