"""Zero-copy host->device staging for MI300A-class unified-memory APUs.

Exposes a tiny, dependency-free API used by the imagenet / minGPT-ddp / FSDP2
benchmarks behind their ``--migrate`` flag:

    from zerocopy import Stager
    stager = Stager(device, enabled=args.migrate)
    host_buf = stager.host_empty((B, C, H, W), torch.float32)  # fill on CPU
    gpu_view = stager.to_device(host_buf)                       # no copy if enabled

Design
------
On a discrete GPU ``tensor.to('cuda')`` DMAs the bytes into a separate device
allocation. On the MI300A APU host and device share the same physical HBM, so
the copy is unnecessary -- ``migrate()`` (in the ``migrate_ext`` HIP extension)
hands the GPU the same pointer and only updates page residency.

Two zero-copy methods are supported (``method=``):
  * ``"managed"``  -- allocate the host buffer in hipMallocManaged memory
    (``host_empty``) and alias it. Lowest per-call cost; requires you to use
    ``host_empty`` for the staging buffer.
  * ``"register"`` -- hipHostRegister an arbitrary, already-allocated (pageable)
    CPU tensor and alias it. Works on tensors you did not allocate (e.g. a
    DataLoader batch); registration is a one-time cost per buffer.

Both avoid the data copy *and* the duplicate device allocation, so the batch
occupies its bytes once in HBM instead of twice (host copy + device copy).

Everything degrades gracefully:
  * If the HIP extension fails to build, or HSA_XNACK!=1, or the device is not a
    unified-memory APU, ``Stager`` falls back to plain ``.to(device)`` and a
    normal pinned/pageable host buffer, so results stay correct everywhere.
"""
import os
import warnings

import torch

_EXT = None
_EXT_ERR = None


def _load_ext():
    """JIT-compile and cache the migrate_ext HIP extension (best effort)."""
    global _EXT, _EXT_ERR
    if _EXT is not None or _EXT_ERR is not None:
        return _EXT
    try:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        _EXT = load(
            name="migrate_ext",
            sources=[os.path.join(here, "migrate_ext.cpp")],
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001 - any build/runtime failure -> fallback
        _EXT_ERR = e
        _EXT = None
    return _EXT


def unified_memory_available():
    """True only if the migrate extension loaded, XNACK is on, and dev is ROCm."""
    if os.environ.get("HSA_XNACK") != "1":
        return False
    if not (torch.cuda.is_available() and torch.version.hip):
        return False
    return _load_ext() is not None


class Stager:
    """Stage host input batches to the GPU, optionally zero-copy via migrate().

    method: "managed" (alias hipMallocManaged buffers) or "register"
            (hipHostRegister arbitrary pageable buffers). Ignored when disabled.
    """

    def __init__(self, device, enabled=True, method="managed", prefetch=True):
        self.device = torch.device(device)
        self.prefetch = prefetch
        self.method = method
        self.enabled = bool(enabled) and unified_memory_available()
        # Cache of host data_ptr -> GPU alias, so a reused staging buffer is
        # migrated/registered once and cheaply reused across steps.
        self._cache = {}
        if enabled and not self.enabled:
            reason = ("HSA_XNACK != 1" if os.environ.get("HSA_XNACK") != "1"
                      else f"migrate_ext unavailable ({_EXT_ERR})")
            warnings.warn(f"--migrate requested but disabled: {reason}; "
                          "falling back to .to(device) copies.")

    @property
    def mode(self):
        return f"migrate:{self.method}(zero-copy)" if self.enabled else "to(copy)"

    def host_empty(self, shape, dtype):
        """A host buffer to fill on the CPU.

        managed method  -> hipMallocManaged memory (aliasable by migrate()).
        register method -> ordinary *pageable* memory; register_migrate() will
                           hipHostRegister it (registering already-pinned memory
                           fails), then alias it in place.
        disabled        -> pinned host memory so the fallback .to() copy is fast.
        """
        if self.enabled and self.method == "managed":
            return _EXT.managed_empty(list(shape), dtype)
        if self.enabled and self.method == "register":
            return torch.empty(shape, dtype=dtype)  # pageable; registered later
        try:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        except Exception:
            return torch.empty(shape, dtype=dtype)

    def to_device(self, host_tensor):
        """Return a GPU tensor for a host buffer: aliased (zero-copy) or copied.

        For a reused buffer the alias is created once and cached; because it
        shares storage with the host tensor, later host writes remain visible.
        """
        if not self.enabled:
            return host_tensor.to(self.device, non_blocking=True)
        key = host_tensor.data_ptr()
        view = self._cache.get(key)
        if view is None:
            if self.method == "register":
                view = _EXT.register_migrate(host_tensor)
            else:
                view = _EXT.migrate(host_tensor, self.prefetch)
            self._cache[key] = view
        return view
