"""Backend selection + the DirectML op substitutes.

The substitutes are exercised here on the CPU (via `BOLTOPT_FORCE_FALLBACKS`
and by calling them directly), so they stay covered in the environment the test
suite normally runs in rather than only in the DirectML venv.
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.backend import (CPU, CUDA, DIRECTML, Backend, available_backends,
                         backend_details, backend_from_settings, eye_bool,
                         is_directml_device, needs_sampler_fallback,
                         resolve_backend)
from app.geometry import (bilinear_sample, build_legal_region, build_region_sdf,
                          trilinear_sample)

RECT = [(0, 0), (200, 0), (200, 100), (0, 100), (0, 0)]


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def test_cpu_is_always_available_and_first():
    avail = available_backends()
    assert avail[0] == CPU
    assert set(avail) <= {CPU, CUDA, DIRECTML}


def test_cpu_backend_is_float64():
    be = resolve_backend(CPU)
    assert be.name == CPU and be.dtype == torch.float64
    assert be.device.type == "cpu"


def test_unavailable_backend_degrades_to_cpu_rather_than_raising():
    """A project saved on another machine must still open here."""
    missing = next(n for n in (CUDA, DIRECTML) if n not in available_backends())
    assert resolve_backend(missing).name == CPU


def test_unknown_name_degrades_to_cpu():
    assert resolve_backend("nonsense").name == CPU
    assert resolve_backend(None).name == CPU


def test_directml_backend_is_float32_when_present():
    if DIRECTML not in available_backends():
        pytest.skip("no DirectML device")
    be = resolve_backend(DIRECTML)
    # not a tuning choice: DirectML has no float64 kernels at all
    assert be.dtype == torch.float32
    assert be.uses_fallbacks


def test_backend_details_keys():
    d = backend_details()
    assert set(d) == {CUDA, DIRECTML}
    for name, detail in d.items():
        assert (detail is None) == (name not in available_backends())


def test_legacy_use_gpu_setting_still_selects_a_backend():
    assert backend_from_settings({"use_gpu": False}).name == CPU
    want = CUDA if CUDA in available_backends() else CPU
    assert backend_from_settings({"use_gpu": True}).name == want
    # an explicit `backend` wins over the legacy flag
    assert backend_from_settings({"backend": "cpu", "use_gpu": True}).name == CPU


def test_backend_tensor_casts_float64_numpy_down():
    be = Backend("fake", torch.device("cpu"), torch.float32)
    t = be.tensor(np.array([[1.5, 2.5]], dtype=np.float64))
    assert t.dtype == torch.float32 and t.device.type == "cpu"
    assert be.tensor([[1, 2]], dtype=torch.long).dtype == torch.long
    assert be.tensor([[1.0, 2.0]], requires_grad=True).requires_grad


def test_is_directml_device():
    assert not is_directml_device(torch.device("cpu"))
    assert is_directml_device("privateuseone:0")


# --------------------------------------------------------------------------
# eye_bool -- torch.eye returns an EMPTY tensor on DirectML
# --------------------------------------------------------------------------

def test_eye_bool_matches_torch_eye_on_cpu():
    for n in (1, 2, 8):
        got = eye_bool(n, torch.device("cpu"))
        assert got.dtype == torch.bool and got.shape == (n, n)
        assert torch.equal(got, torch.eye(n, dtype=torch.bool))


def test_eye_bool_is_cached_per_device():
    a = eye_bool(6, torch.device("cpu"))
    b = eye_bool(6, torch.device("cpu"))
    assert a is b


# --------------------------------------------------------------------------
# grid_sample substitutes
# --------------------------------------------------------------------------

def test_force_fallbacks_env(monkeypatch):
    monkeypatch.setenv("BOLTOPT_FORCE_FALLBACKS", "1")
    assert needs_sampler_fallback(torch.device("cpu"))
    monkeypatch.setenv("BOLTOPT_FORCE_FALLBACKS", "0")
    assert not needs_sampler_fallback(torch.device("cpu"))


def test_bilinear_sample_matches_grid_sample():
    g = torch.Generator().manual_seed(3)
    vol = torch.randn(37, 53, dtype=torch.float64, generator=g)
    gx = torch.rand(400, dtype=torch.float64, generator=g) * 2 - 1
    gy = torch.rand(400, dtype=torch.float64, generator=g) * 2 - 1
    ref = F.grid_sample(vol[None, None], torch.stack([gx, gy], -1).view(1, 1, -1, 2),
                        mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0]
    assert torch.allclose(bilinear_sample(vol, gx, gy), ref, atol=1e-12)


def test_bilinear_sample_clamps_like_padding_border():
    vol = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    out = bilinear_sample(vol, torch.tensor([-3.0, 3.0]), torch.tensor([-3.0, 3.0]))
    assert out.tolist() == [0.0, 11.0]


def test_trilinear_sample_matches_grid_sample():
    g = torch.Generator().manual_seed(5)
    vol = torch.randn(9, 17, 23, dtype=torch.float64, generator=g)
    c = torch.rand(3, 500, dtype=torch.float64, generator=g) * 2 - 1
    gx, gy, gz = c[0], c[1], c[2]
    grid = torch.stack([gx, gy, gz], -1).view(1, 1, 1, -1, 3)
    ref = F.grid_sample(vol[None, None], grid, mode="bilinear",
                        padding_mode="border", align_corners=True)[0, 0, 0, 0]
    assert torch.allclose(trilinear_sample(vol, gx, gy, gz), ref, atol=1e-12)


def test_trilinear_nearest_matches_grid_sample():
    g = torch.Generator().manual_seed(7)
    vol = (torch.rand(8, 11, 13, dtype=torch.float64, generator=g) > 0.5).double()
    c = torch.rand(3, 300, dtype=torch.float64, generator=g) * 2 - 1
    grid = torch.stack([c[0], c[1], c[2]], -1).view(1, 1, 1, -1, 3)
    ref = F.grid_sample(vol[None, None], grid, mode="nearest",
                        padding_mode="border", align_corners=True)[0, 0, 0, 0]
    got = trilinear_sample(vol, c[0], c[1], c[2], nearest=True)
    # both are 0/1 labels; exact agreement away from the half-cell ties
    assert (got == ref).double().mean().item() > 0.98


def test_fallback_sampler_gradient_matches_grid_sample():
    g = torch.Generator().manual_seed(11)
    vol = torch.randn(31, 41, dtype=torch.float64, generator=g)
    base = (torch.rand(50, 2, dtype=torch.float64, generator=g) * 1.6 - 0.8)

    a = base.clone().requires_grad_(True)
    bilinear_sample(vol, a[:, 0], a[:, 1]).sum().backward()

    b = base.clone().requires_grad_(True)
    F.grid_sample(vol[None, None], b.view(1, 1, -1, 2), mode="bilinear",
                  padding_mode="border", align_corners=True).sum().backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-9)


# --------------------------------------------------------------------------
# End to end: the fallback path must give the same fields as the fast path
# --------------------------------------------------------------------------

def _fields():
    region = build_legal_region(RECT, [], 10.0)
    material = Polygon([(0, 0), (200, 0), (200, 100), (0, 100)],
                       [[(90, 40), (110, 40), (110, 60), (90, 60)]])
    return build_region_sdf(region, resolution=192, material_region=material,
                            n_dirs=16, ray_resolution=96, ray_cap=30.0)


def test_region_fields_fallback_sampling_agrees(monkeypatch):
    # pin the flag rather than inherit it: this test compares the two paths
    # against each other, so it has to run both however the env is set.
    monkeypatch.setenv("BOLTOPT_FORCE_FALLBACKS", "0")
    f = _fields()
    g = torch.Generator().manual_seed(13)
    pts = torch.rand(200, 2, dtype=torch.float64, generator=g) * torch.tensor([200.0, 100.0])
    ang = torch.rand(200, dtype=torch.float64, generator=g) * 6.28

    assert not f.fallback_sampling
    sd_fast, ray_fast = f.sample(pts), f.ray_distance(pts, ang)

    monkeypatch.setenv("BOLTOPT_FORCE_FALLBACKS", "1")
    f2 = _fields()
    assert f2.fallback_sampling
    assert torch.allclose(f2.sample(pts), sd_fast, atol=1e-10)
    assert torch.allclose(f2.ray_distance(pts, ang), ray_fast, atol=1e-10)


def test_to_preserves_pooled_ray_volume():
    f = _fields()
    moved = f.to(device=torch.device("cpu"), dtype=torch.float32)
    assert moved.dtype == torch.float32
    assert moved._r.shape == f._r.shape
    assert torch.allclose(moved._r.double(), f._r, atol=1e-3)
