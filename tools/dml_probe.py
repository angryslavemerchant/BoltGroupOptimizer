"""Probe which torch ops the optimizer needs actually work on a given device.

Run it in the DirectML venv to find out what DirectML supports before trusting
the optimizer to it:

    .venv-dml\\Scripts\\python.exe tools\\dml_probe.py --backend directml

Every check mirrors a real call site in `app/optimizer.py` / `app/geometry.py`
(shapes and dtypes included), runs forward and, where it matters, backward.
Prints one row per op: OK / FAIL (with the exception) plus the wall time, so a
silent CPU fallback shows up as a suspiciously slow OK rather than as nothing.
"""
import argparse
import math
import sys
import time
import traceback
import warnings

import torch

sys.path.insert(0, __file__.rsplit("tools", 1)[0])
from app.backend import available_backends, resolve_backend  # noqa: E402


ROWS = []


def _assert_eye(n, dev):
    e = torch.eye(n, dtype=torch.bool, device=dev)
    assert tuple(e.shape) == (n, n), f"returned {tuple(e.shape)}, expected ({n}, {n})"


def _assert_eye_cpu(n, dev):
    e = torch.eye(n, dtype=torch.bool).to(dev)
    assert tuple(e.shape) == (n, n) and int(e.cpu().sum()) == n


def check(name, fn):
    """Run one op, recording OK / FAIL / 'OK (cpu fallback)'.

    The third outcome is the one that matters most and the one that is easiest
    to miss: torch_directml does not raise on an op it lacks a kernel for, it
    quietly copies the whole call to the CPU and back and emits a one-shot
    UserWarning.  That is functionally correct and catastrophically slow in an
    inner loop, so the warning is captured per check rather than left to be
    printed once for the whole run.
    """
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fn()
            ms = f"{(time.perf_counter() - t0) * 1e3:.1f} ms"
            fell = [w for w in caught if "fall back to run on the CPU" in str(w.message)]
            if fell:
                op = str(fell[0].message).split("'")[1] if "'" in str(fell[0].message) else "?"
                ROWS.append((name, "CPU-FB", ms, f"no DML kernel for {op}; silently runs on CPU"))
            else:
                ROWS.append((name, "OK", ms, ""))
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:100] if str(e) else type(e).__name__
            ROWS.append((name, "FAIL", f"{(time.perf_counter() - t0) * 1e3:.1f} ms", msg))
            if "-v" in sys.argv:
                traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="directml")
    args = ap.parse_args()

    # torch_directml announces a CPU fallback with TORCH_WARN_ONCE, which is
    # *once per process* -- so without this the second op that falls back looks
    # like a clean OK. Turning it into a plain warning makes every fallback
    # visible on the row that caused it.
    torch.set_warn_always(True)
    print("available backends:", available_backends())
    be = resolve_backend(args.backend)
    dev, dt = be.device, be.dtype
    print(f"probing backend={be.name} device={dev} dtype={dt}\n")

    B, C, N, K = 64, 2, 8, 32
    ny, nx = 96, 128

    def T(*shape):
        return torch.randn(*shape, device=dev, dtype=dt)

    # --- float64 support (the reason dtype is a backend property at all) ----
    def f64():
        x = torch.randn(4, 4, device=dev, dtype=torch.float64)
        (x @ x).cpu()
    check("float64 tensors", f64)

    # --- geometry: grid_sample ---------------------------------------------
    import torch.nn.functional as Fn

    def gs4():
        g = torch.rand(1, 1, B * N, 2, device=dev, dtype=dt) * 2 - 1
        vol = T(1, 1, ny, nx)
        Fn.grid_sample(vol, g, mode="bilinear", padding_mode="border",
                       align_corners=True).cpu()
    check("grid_sample 4-D bilinear border", gs4)

    def gs5():
        g = torch.rand(1, 1, 1, B * C * N, 3, device=dev, dtype=dt) * 2 - 1
        vol = T(1, 1, K, ny, nx)
        Fn.grid_sample(vol, g, mode="bilinear", padding_mode="border",
                       align_corners=True).cpu()
    check("grid_sample 5-D trilinear border", gs5)

    def gs5n():
        g = torch.rand(1, 1, 1, B * C * N, 3, device=dev, dtype=dt) * 2 - 1
        vol = T(1, 1, K, ny, nx)
        Fn.grid_sample(vol, g, mode="nearest", padding_mode="border",
                       align_corners=True).cpu()
    check("grid_sample 5-D nearest (ray kind)", gs5n)

    def gs4_bwd():
        p = torch.zeros(1, 1, B * N, 2, device=dev, dtype=dt, requires_grad=True)
        vol = T(1, 1, ny, nx)
        Fn.grid_sample(vol, p, mode="bilinear", padding_mode="border",
                       align_corners=True).sum().backward()
        assert p.grad is not None and torch.isfinite(p.grad).all().item()
    check("grid_sample 4-D backward", gs4_bwd)

    def gs5_bwd():
        p = torch.zeros(1, 1, 1, B * N, 3, device=dev, dtype=dt, requires_grad=True)
        vol = T(1, 1, K, ny, nx)
        Fn.grid_sample(vol, p, mode="bilinear", padding_mode="border",
                       align_corners=True).sum().backward()
        assert p.grad is not None and torch.isfinite(p.grad).all().item()
    check("grid_sample 5-D backward", gs5_bwd)

    # --- objective ----------------------------------------------------------
    def lse():
        x = T(B, C * N)
        m = torch.rand(B, C * N, device=dev) > 0.3
        neg = torch.where(m, torch.zeros_like(x), torch.full_like(x, -1e18))
        torch.logsumexp(x + neg, dim=-1).cpu()
    check("logsumexp (masked -1e18)", lse)

    def lse_inf():
        x = T(B, C * N)
        torch.logsumexp(torch.cat([x, torch.full_like(x[:, :1], -float("inf"))], -1), -1).cpu()
    check("logsumexp with -inf entry", lse_inf)

    def maxinf():
        x = T(B, C * N)
        m = torch.rand(B, C * N, device=dev) > 0.3
        torch.where(m, x, torch.full_like(x, -float("inf"))).max(dim=-1).values.cpu()
    check("max over -inf-masked tensor", maxinf)

    def cd():
        a = T(B, N, 2)
        torch.cdist(a, a).cpu()
    check("cdist", cd)

    def cd_bwd():
        a = T(B, N, 2).requires_grad_(True)
        torch.cdist(a, a).sum().backward()
        assert torch.isfinite(a.grad).all().item()
    check("cdist backward", cd_bwd)

    def at2():
        a, b = T(B, C, N), T(B, C, N)
        torch.atan2(a, b).cpu()
    check("atan2", at2)

    def at2_bwd():
        a = T(B, C, N).requires_grad_(True)
        b = T(B, C, N)
        torch.atan2(a, b).sum().backward()
        assert torch.isfinite(a.grad).all().item()
    check("atan2 backward", at2_bwd)

    def rem():
        torch.remainder(T(B, C, N), 2 * math.pi).cpu()
    check("remainder", rem)

    for nm, fn in [
        ("sqrt", lambda: torch.sqrt(T(B, N).abs() + 1e-9).cpu()),
        ("clamp", lambda: torch.clamp(T(B, N), min=0.0, max=1.0).cpu()),
        ("where", lambda: torch.where(T(B, N) > 0, T(B, N), T(B, N)).cpu()),
        ("minimum", lambda: torch.minimum(T(B, N), T(B, N)).cpu()),
        ("triu bool mask", lambda: (T(B, N, N) * torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=dev), 1).to(dt)).sum().cpu()),
        # NB: the assert is the point -- torch.eye does not raise on DirectML,
        # it returns an EMPTY tensor, which would silently gut the ~eye masks.
        ("torch.eye(device=) shape", lambda: _assert_eye(N, dev)),
        ("eye built on cpu then .to(device)", lambda: _assert_eye_cpu(N, dev)),
        ("torch.full(py float, no dtype)", lambda: torch.full((3, 3), 2.5, device=dev).cpu()),
        ("bool mask indexing P[m]", lambda: T(B, N, 2)[0][
            torch.rand(N, device=dev) > 0.5].cpu()),
        ("index_select (gather rows)", lambda: T(B, N, 2).index_select(
            0, torch.arange(0, B, 2, device=dev)).cpu()),
        ("gather dim1", lambda: T(B, C, N).gather(
            1, torch.zeros(B, 1, N, dtype=torch.long, device=dev)).cpu()),
        ("argsort", lambda: torch.argsort(T(B)).cpu()),
        ("topk", lambda: torch.topk(T(B), 8, largest=False).indices.cpu()),
        ("unique (cull, long)", lambda: torch.unique(
            torch.randint(2, 9, (B,), device=dev)).cpu()),
        ("nonzero", lambda: torch.nonzero(T(B) > 0, as_tuple=False).flatten().cpu()),
        ("argmin/argmax", lambda: (torch.argmin(T(B)).cpu(), torch.argmax(T(B)).cpu())),
        ("min(dim).values", lambda: T(B, C, N, N).min(dim=-1).values.cpu()),
        ("isfinite", lambda: torch.isfinite(T(B)).all().cpu()),
        ("expand+stack", lambda: torch.stack(
            [-T(B, N, 2)[..., 1], T(B, N, 2)[..., 0]], -1).cpu()),
    ]:
        check(nm, fn)

    def reduce_bwd(fn):
        def go():
            x = T(B, C, N, N).requires_grad_(True)
            fn(x).sum().backward()
            assert torch.isfinite(x.grad).all().item()
        return go
    check("min(dim).values BACKWARD", reduce_bwd(lambda x: x.min(dim=-1).values))
    check("amin BACKWARD", reduce_bwd(lambda x: torch.amin(x, dim=-1)))
    check("max(dim).values BACKWARD", reduce_bwd(lambda x: x.max(dim=-1).values))
    check("gather BACKWARD", reduce_bwd(
        lambda x: x.gather(-1, x.detach().argmin(-1, keepdim=True)).squeeze(-1)))

    def setitem():
        P = T(B, N, 2)
        with torch.no_grad():
            P[:, :2] = torch.zeros(2, 2, device=dev, dtype=dt)
        P.cpu()
    check("in-place slice assign P[:, :F]", setitem)

    def adam():
        P = T(B, N, 2).requires_grad_(True)
        opt = torch.optim.Adam([P], lr=0.1)
        for _ in range(3):
            opt.zero_grad()
            (P * P).sum().backward()
            P.grad[:, :1] = 0.0
            opt.step()
        assert torch.isfinite(P.detach()).all().item()
    check("Adam step (3 iters)", adam)

    def full_loss():
        """The whole objective, end to end, through the real code."""
        from app.geometry import RegionFields
        grid = T(ny, nx)
        ray = T(K, 64, 64)
        kind = (torch.rand(K, 64, 64, device=dev, dtype=dt) > 0.5).to(dt)
        f = RegionFields(grid, 0.0, 0.0, 100.0, 80.0, ray=ray, ray_kind=kind)
        from app.optimizer import batched_loss
        P = (torch.rand(B, N, 2, device=dev, dtype=dt) * 50 + 10).requires_grad_(True)
        M = torch.ones(B, N, dtype=torch.bool, device=dev)
        fp = torch.tensor([[60.0, 40.0]], device=dev, dtype=dt)
        fv = torch.tensor([[500.0, -300.0]], device=dev, dtype=dt)
        loss, _ = batched_loss(P, M, fp, fv, f, 10.0,
                               bearing={"fields": f, "d": 6.0, "k_min": 0.05})
        loss.sum().backward()
        assert torch.isfinite(P.grad).all().item(), "non-finite gradient"
    check("full batched_loss fwd+bwd (bearing on)", full_loss)

    w = max(len(r[0]) for r in ROWS) + 2
    print(f"{'op':<{w}}{'status':<8}{'time':<11}note")
    print("-" * (w + 30))
    for name, st, tm, note in ROWS:
        print(f"{name:<{w}}{st:<8}{tm:<11}{note}")
    print()
    print(f"{sum(1 for r in ROWS if r[1] == 'OK')} OK, "
          f"{sum(1 for r in ROWS if r[1] == 'CPU-FB')} silent CPU fallback, "
          f"{sum(1 for r in ROWS if r[1] == 'FAIL')} failing, of {len(ROWS)}")


if __name__ == "__main__":
    main()
