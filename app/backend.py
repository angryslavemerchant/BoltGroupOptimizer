"""Compute backend selection: cpu / cuda / directml.

Everything that builds a tensor goes through one `Backend`, which carries the
three things that differ between them:

* ``device`` -- where tensors live;
* ``dtype``  -- float64 on cpu/cuda, **float32 on directml**, because DirectML
  has no float64 kernels at all (a `torch.float64` tensor cannot even be moved
  onto the device).  Precision is therefore a property of the backend, not a
  constant, and every `torch.tensor(...)` call site takes it from here;
* ``uses_fallbacks`` -- whether the hand-written substitutes for ops DirectML
  is missing are in play (see `app/geometry.py` / `app/optimizer.py`).

DirectML lives in its own venv (see `requirements-directml.txt`) because
`torch_directml` hard-pins the torch it was built against, so importing it has
to stay optional and failure-tolerant here.
"""
from dataclasses import dataclass, field
import os
import torch

CPU, CUDA, DIRECTML = "cpu", "cuda", "directml"

# Set BOLTOPT_FORCE_FALLBACKS=1 to exercise the DirectML fallback code paths on
# an ordinary CPU/CUDA backend, so they keep being covered by the test suite
# instead of rotting in an environment nobody runs tests in.
_FORCE_FALLBACKS_ENV = "BOLTOPT_FORCE_FALLBACKS"


def _force_fallbacks_env():
    return os.environ.get(_FORCE_FALLBACKS_ENV, "").strip() not in ("", "0", "false", "False")


@dataclass(frozen=True)
class Backend:
    name: str
    device: torch.device
    dtype: torch.dtype
    uses_fallbacks: bool = False
    detail: str = ""

    @property
    def is_directml(self):
        return self.name == DIRECTML

    def tensor(self, data, dtype=None, requires_grad=False):
        """`torch.tensor` with this backend's device/dtype applied.

        Note the two-step construction: DirectML cannot take a float64 numpy
        array directly, and `torch.tensor(np_f64, dtype=torch.float32,
        device=dml)` still materializes a float64 intermediate on some paths, so
        the cast happens on the CPU first and only the final dtype is moved.
        """
        dt = dtype if dtype is not None else self.dtype
        t = torch.as_tensor(data).to(dtype=dt).to(device=self.device)
        return t.requires_grad_(True) if requires_grad else t

    def zeros(self, *shape, dtype=None):
        return torch.zeros(*shape, dtype=dtype or self.dtype, device=self.device)

    def describe(self):
        return f"{self.name}" + (f" ({self.detail})" if self.detail else "")


_dml_probe_cache = {}


def _directml_module():
    try:
        import torch_directml  # type: ignore
        return torch_directml
    except Exception:
        return None


def directml_status():
    """(available, adapter_name_or_None). Cached: probing allocates a device."""
    if "status" in _dml_probe_cache:
        return _dml_probe_cache["status"]
    status = (False, None)
    mod = _directml_module()
    if mod is not None:
        try:
            if mod.is_available() and mod.device_count() > 0:
                dev = mod.device()
                # A device handle alone proves nothing -- run something on it.
                a = torch.ones(4, 4, dtype=torch.float32, device=dev)
                float((a @ a).sum().cpu())
                name = None
                try:
                    # device_name comes back straight from the DXGI adapter
                    # description, NUL-terminated and space-padded -- the raw
                    # string is not JSON-safe, so trim both.
                    name = str(mod.device_name(0)).replace("\x00", "").strip()
                except Exception:
                    name = "DirectML device"
                name = name or "DirectML device"
                status = (True, name)
        except Exception:
            status = (False, None)
    _dml_probe_cache["status"] = status
    return status


def cuda_status():
    try:
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return False, None


def available_backends():
    """Backend names this process can actually run on. `cpu` is always first."""
    out = [CPU]
    if cuda_status()[0]:
        out.append(CUDA)
    if directml_status()[0]:
        out.append(DIRECTML)
    return out


def backend_details():
    return {CUDA: cuda_status()[1], DIRECTML: directml_status()[1]}


def resolve_backend(name=None, force_fallbacks=None):
    """Name -> `Backend`, silently falling back to cpu when it is unavailable.

    Unavailable is not an error: a saved project that asks for `cuda` should
    still open on a laptop without one, so the request degrades to cpu and the
    result payload reports which backend actually ran.
    """
    name = (name or CPU).strip().lower()
    if name in ("gpu", "dml", "direct-ml"):
        name = DIRECTML if directml_status()[0] else CUDA
    ff = _force_fallbacks_env() if force_fallbacks is None else bool(force_fallbacks)

    if name == CUDA and cuda_status()[0]:
        return Backend(CUDA, torch.device("cuda"), torch.float64, ff, cuda_status()[1] or "")
    if name == DIRECTML:
        ok, adapter = directml_status()
        if ok:
            dev = _directml_module().device()
            # float32 is not a tuning choice here: DirectML has no float64 kernels.
            return Backend(DIRECTML, dev, torch.float32, True, adapter or "")
    return Backend(CPU, torch.device("cpu"), torch.float64, ff, "")


# ---------------------------------------------------------------------------
# Op substitutes for DirectML gaps.
#
# These are deliberately *not* gated on the backend object: they are keyed on a
# device (or are unconditional), so any tensor that wanders onto a DirectML
# device gets the right behaviour without having to thread a Backend through
# every helper.  `BOLTOPT_FORCE_FALLBACKS=1` turns them on everywhere, which is
# how the CPU test suite keeps them covered.
# ---------------------------------------------------------------------------

def is_directml_device(device):
    """torch_directml devices report as `privateuseone`, not as a named type."""
    try:
        return torch.device(device).type in ("privateuseone", "dml")
    except Exception:
        return False


def needs_sampler_fallback(device):
    """True when `F.grid_sample` must be replaced by the hand-written samplers.

    DirectML has no `grid_sampler_2d`/`_3d` kernel; torch_directml does not
    error, it transparently copies the whole call to the CPU and back, which in
    an inner loop is far worse than a slightly clumsier on-device implementation.
    """
    return _force_fallbacks_env() or is_directml_device(device)


_eye_cache = {}


def eye_bool(n, device):
    """A boolean identity matrix that is correct on DirectML.

    `torch.eye(n, device=dml)` does not raise -- it returns an **empty** tensor,
    so the `~eye` self-exclusion masks in the bearing and spacing code silently
    became zero-sized and the subsequent view blew up (or, worse, could have
    broadcast).  Building on the CPU and copying is exact everywhere; n is the
    bolt count, so the copy is a few hundred bytes, and the result is cached per
    (n, device) anyway.
    """
    key = (int(n), str(device))
    t = _eye_cache.get(key)
    if t is None:
        t = torch.eye(int(n), dtype=torch.bool).to(device)
        _eye_cache[key] = t
    return t


def backend_from_settings(settings):
    """Read the `backend` setting (with back-compat for the old `use_gpu` bool)."""
    name = settings.get("backend")
    if not name:
        name = CUDA if settings.get("use_gpu") else CPU
    return resolve_backend(name)
