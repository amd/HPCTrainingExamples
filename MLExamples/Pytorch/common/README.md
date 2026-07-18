# Zero-copy host→GPU input staging on MI300A (`migrate` vs `.to`)

Shared helpers for the `imagenet`, `minGPT-ddp`, and `FSDP2` benchmarks that
exploit the **MI300A APU's unified, coherent HBM**: the CPU and GPU address the
*same* physical memory, so moving a host tensor onto the GPU does **not** require
copying the bytes — it only needs to hand the GPU the same pointer (and,
optionally, update page residency).

| File | Purpose |
|------|---------|
| `migrate_ext.cpp` | HIP/torch C++ extension: `managed_empty()`, `migrate()`, `register_migrate()` |
| `zerocopy.py` | `Stager` helper (methods `managed`/`register`) with a safe fallback to `.to(device)` |
| `migrate_bench.py` | Micro-benchmark: `.to` copy vs both zero-copy methods, per size, + HBM footprint |

## The idea

On a **discrete** GPU, `tensor.to('cuda')` allocates a separate device buffer and
`hipMemcpy`s the bytes across the bus — an unavoidable O(bytes) transfer. On the
**MI300A** APU there is one pool of HBM shared coherently by the CPU and GPU, so
that copy is physically redundant. Two zero-copy methods avoid it:

**`managed`** (`--migrate-method managed`, default):

1. allocate the host staging buffer in **`hipMallocManaged`** memory
   (`managed_empty`), so the exact same virtual address is dereferenceable from
   device kernels, and
2. return a **CUDA tensor that aliases the same pointer** (`torch::from_blob`) —
   no data copy — optionally issuing `hipMemPrefetchAsync` to mark the pages
   device-resident.

**`register`** (`--migrate-method register`):

1. take **any ordinary, already-allocated (pageable)** CPU tensor — no need to
   pre-allocate it in managed memory — and **`hipHostRegister`** its pages for
   device access, then
2. return a CUDA tensor aliasing the mapped device pointer
   (`hipHostGetDevicePointer`). The pages are `hipHostUnregister`ed when the view
   is freed.

Use `managed` when you control the staging allocation (lowest per-call cost);
use `register` to migrate a tensor produced by code you do not control (e.g. a
`DataLoader` batch) in place. Both are the ML-training analogue of the CG-GPU
solver's `staged_unified` / `alltoallv_unified` host-path variants: leverage
unified memory to turn a copy into a metadata operation.

> **`register` needs pageable memory.** `hipHostRegister` fails on memory that is
> already pinned (`hipHostMalloc`), so the `register` staging buffer must be a
> plain `torch.empty(...)` (the `Stager` handles this). The `managed` method's
> buffer must come from `managed_empty`.

## Safety / portability

`Stager` degrades gracefully. Zero-copy is used **only** when all of the
following hold; otherwise it silently falls back to a pinned-host buffer plus
`.to(device)`, so results are identical everywhere:

* the `migrate_ext` extension builds (needs the ROCm PyTorch toolchain), and
* `HSA_XNACK=1` is set, and
* the device is a ROCm/HIP GPU.

Enable it in the benchmarks with `--migrate` (pick the method with
`--migrate-method managed|register`); compare against an explicit copy baseline
with `--host-copy`. With none of these flags the input batch stays pre-resident
on the GPU (the historical default), so previously published numbers are intact.

## Measured raw transfer cost (MI300A, ROCm 7.2.3, PyTorch 2.12, `HSA_XNACK=1`)

`python common/migrate_bench.py` — per-call cost to make a freshly-produced host
buffer usable on the GPU (numerics verified equal in every row). `managed` is the
steady-state (reused-buffer) alias cost; `register` includes `hipHostRegister`:

| bytes | `.to` copy | `migrate:managed` | `migrate:register` | speedup vs copy |
|------:|-----------:|------------------:|-------------------:|----------------:|
| 4 MB   | 0.12 ms (34 GB/s) | 0.004 ms | 0.025 ms | ~5–30× |
| 16 MB  | 0.41 ms (41 GB/s) | 0.004 ms | 0.026 ms | ~16–100× |
| 64 MB  | 1.33 ms (51 GB/s) | 0.005 ms | 0.026 ms | ~50–280× |
| 256 MB | 4.82 ms (56 GB/s) | 0.008 ms | 0.032 ms | ~150–600× |
| 1 GB   | 18.4 ms (58 GB/s) | 0.008 ms | 0.031 ms | ~590–2300× |

The `.to` path is capped at the ~56 GB/s copy-engine rate and scales with size;
both zero-copy methods are ~constant because they never DMA user data — `managed`
is pure `from_blob` metadata (a reused buffer is prefetched once, then cached),
and `register` adds a one-time page registration.

## Measured HBM footprint / memory savings

`migrate_bench.py` also reports the device-resident HBM consumed to stage one
batch (via `torch.cuda.mem_get_info`). The `.to` path keeps a **second, device-
resident copy** of every staged batch; the zero-copy methods keep a **single**
physical copy and alias it:

| batch bytes | `.to` (device copy) | `migrate:managed` | `migrate:register` | device duplicate saved |
|------------:|--------------------:|------------------:|-------------------:|-----------------------:|
| 64 MB   | 67,108,864  | 0 | 0 | **100%** |
| 256 MB  | 268,435,456 | 0 | 0 | **100%** |
| 1 GB    | 1,073,741,824 | 0 | 0 | **100%** |

So each staged batch costs **its full size again in device HBM** under `.to`,
and **zero** extra under `migrate` — the device-side duplicate is eliminated
entirely. The saving scales with the amount of host data in flight:

* a double-buffered / prefetching input pipeline holding *P* batches in flight
  saves *P × batch_bytes* of HBM, and
* it matters most for **large per-sample inputs** (high-resolution images, video
  clips, long-context token+embedding tensors) on the APU's shared 128 GB pool,
  where a duplicated multi-hundred-MB batch (times prefetch depth times ranks)
  competes directly with model/activation memory.

## End-to-end effect in the training benchmarks

| Example | Host input / step | `--host-copy` | `--migrate` |
|---------|-------------------|--------------:|------------:|
| imagenet (resnet50, b=128) | 77 MB image batch | 3934 img/s | 3917 img/s |
| minGPT-ddp (b=8, block=512) | 2 MB token ids | 121665 tok/s | 121920 tok/s |
| FSDP2 (b=8, seq=512) | 2 MB token ids | 91017 tok/s | 90912 tok/s |

(all within run-to-run noise of the `stage_input=none` default: 4000 / 122263 /
91005)

**Why the end-to-end change is small even though the raw copy is ~30× cheaper:**

* These steps are **compute-bound**, and a reused pinned buffer lets the `.to`
  DMA **overlap** the previous step's compute, so its cost is largely hidden.
* The transformer inputs are **token IDs (~2 MB)** — far too small to matter;
  the copy is in the noise regardless.

(`--migrate-method register` gives the same end-to-end numbers as `managed`
within noise; it is validated on all three examples.)

Where `migrate` is a real win in practice:

* **Large, per-step-fresh host tensors** that cannot overlap (e.g. a real image
  `DataLoader` producing a new 77 MB+ batch each step with no double-buffering) —
  the micro-benchmark's 100×–1000× is the ceiling you recover.
* **Memory footprint**: aliasing avoids the *second* (device) copy of the buffer
  entirely (measured 100% of the device duplicate, above) — often the deciding
  factor on the APU where host and device share one 128 GB pool.

## Using the helper in your own code

```python
import sys, torch
sys.path.insert(0, ".../MLExamples/Pytorch/common")
from zerocopy import Stager

# method="managed" (default) or "register"; falls back to .to if XNACK/ext missing
stager = Stager("cuda", enabled=True, method="managed")
host = stager.host_empty((B, C, H, W), torch.float32)   # managed or pageable buffer
host.copy_(produce_batch_on_cpu())             # fill on the host
gpu = stager.to_device(host)                   # zero-copy alias (or .to copy)
```

To migrate a tensor you did **not** allocate (e.g. straight from a DataLoader),
use the `register` method — no managed pre-allocation required:

```python
stager = Stager("cuda", enabled=True, method="register")
for cpu_batch, target in loader:               # cpu_batch is an ordinary tensor
    gpu_batch = stager.to_device(cpu_batch)     # hipHostRegister + alias, no copy
    ...
```

Run with `HSA_XNACK=1` and the ROCm PyTorch module loaded so the extension can
JIT-compile (set `TORCH_EXTENSIONS_DIR=/tmp/$USER/torchext` for a writable cache).
