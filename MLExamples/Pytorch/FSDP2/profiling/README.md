# FSDP2 — per-profiler hands-on guides (RCCL focus)

This directory holds **one hands-on guide per profiler** for the FSDP2 example,
each written around the thing that makes FSDP2 different from DDP: the **sharded
RCCL collectives** — the parameter **all-gather** (forward + backward) and the
gradient **reduce-scatter** (backward). Every guide follows the same recipe:
load the tool, take a **baseline** measurement, apply **one** RCCL/compute lever
from [`../README_rccl_optimization.md`](../README_rccl_optimization.md), and *see
the change in the profiler*. Each page lists the exact command for this workload,
the **text output** to expect, the **screenshots** of the result, and a short set
of **participant exercises**.

This refines the combined hub [`PROFILING.md`](PROFILING.md) into focused,
copy-pasteable walkthroughs. `PROFILING.md` keeps the shared *Which profiler* /
*Ground rules* material and links here.

> These are FSDP2-specific. The **shared, cross-example** command references
> (used by `imagenet`, `minGPT-ddp`, and FSDP2 together) still live under
> [`../../common/profilers/`](../../common/profilers/README.md); the DeepSpeed
> FlopsProfiler and Score-P pages there have no RCCL signal and are not repeated
> here.

## Guides

| Profiler | RCCL signal it surfaces | Lever it shows best | Page |
|----------|-------------------------|---------------------|------|
| **torch.profiler (Kineto)** | `nccl:_all_gather_base` / `nccl:_reduce_scatter_base` op rows (the collectives fuse to one `ncclDevKernel_Generic_2` kernel) | bf16 params/grads (byte volume) | [`torch-profiler.md`](torch-profiler.md) |
| **rocprofv3** | fused `ncclDevKernel_Generic_2` per-kernel total (all RCCL, ~44 % here) + Perfetto timeline | `NCCL_ALGO` / `NCCL_PROTO` | [`rocprofv3.md`](rocprofv3.md) |
| **rocprofiler-systems** | host+GPU Perfetto timeline: all-gather **prefetch overlap** with compute | `--explicit-prefetching` | [`rocprofiler-systems.md`](rocprofiler-systems.md) |
| **TensorBoard** | GUI over the Kineto trace: step-time **communication vs compute** split, overlap | prefetch / bf16 (visual) | [`tensorboard.md`](tensorboard.md) |
| **TraceLens** | automated report: per-collective **algo/bus bandwidth + inter-rank skew**; per-op roofline | any lever (before/after compare) | [`tracelens.md`](tracelens.md) |

`torch.profiler` is the first stop (it speaks the framework's language and needs
no extra tooling). Drop to `rocprofv3` for kernel-level detail, `rocprof-sys` to
*see* the prefetch overlap, TensorBoard for a GUI, and **TraceLens** to turn any
of those traces into a quantified RCCL bandwidth/skew table without hand-reading
Perfetto.

## Ground rules (or your numbers are noise)

Everything in the [quick-start `README.md`](../README.md) and the
[benchmark guide](../benchmarks/README_benchmark.md) applies **more** under a
profiler, because profiling adds overhead:

- Profile inside an affinity-bound, `--exclusive` allocation with **≥2 GPUs**
  (FSDP2 shards across ranks; 1 GPU is not a sharded run). 4+ GPUs gives a
  clearer RCCL signal.
- Nodes must be booted with **`iommu=pt`** so RCCL uses direct xGMI P2P
  (`grep -o iommu=pt /proc/cmdline`). `fsdp2_bench.py` already passes `device_id=`
  to `init_process_group` (required — see the
  [benchmark "Cluster fixes"](../benchmarks/README_benchmark.md#running-on-mi300a-required-settings-and-cluster-fixes)).
- Profile a **fixed, short window** (`--profile` uses wait=1/warmup=3/active=6).
- Keep the **per-rank** traces separate (`--profile` writes `rank0/`, `rank1/`, …).
- **bf16 hipBLASLt stall (ROCm 7.2.x):** the bf16 path can hang for minutes in
  hipBLASLt. Run on ROCm 6.4.3, or `export TORCH_BLAS_PREFER_HIPBLASLT=0`. See
  [`../../common/hipblaslt-notes.md`](../../common/hipblaslt-notes.md).
- **RCCL 2.27 fuses the collectives (this stack).** Every collective runs as one
  device kernel, `ncclDevKernel_Generic_2` — there are **no** separate
  `ncclDevKernel_AllGather_*` / `…_ReduceScatter_*` rows, and search terms /
  name-based classifiers must use `ncclDevKernel_Generic`. To split all-gather vs.
  reduce-scatter, read the framework op rows (torch.profiler) or
  [TraceLens](tracelens.md). Raw device-kernel time is also **skew-inflated** —
  TraceLens separates the true transport from inter-rank wait.

## The communication ceiling (rccl-tests / TransferBench)

A profiler tells you the **achieved** all-gather/reduce-scatter bandwidth for
*this* workload. To know whether that is good, compare it against the fabric's
**achievable** ceiling — the communication analogue of a compute roofline. Two
microbenchmarks give it:

```bash
# rccl-tests: the collective RCCL actually runs for FSDP2
#   (module or build from https://github.com/ROCm/rccl-tests)
mpirun -np 8 ./build/all_gather_perf     -b 8M -e 1G -f 2 -g 1
mpirun -np 8 ./build/reduce_scatter_perf -b 8M -e 1G -f 2 -g 1
# -> the "busbw" column at your message size is the ceiling; TraceLens'
#    measured "bus bw (GB/s)" for the same collective should approach it.

# TransferBench: raw device-to-device xGMI/PCIe link bandwidth (point-to-point)
./TransferBench a2a 64M            # all-to-all pairs at 64 MB
```

Read the profiler's per-collective bus bandwidth (see [`tracelens.md`](tracelens.md))
next to these numbers: a large gap means the run is latency-bound or skewed
(tune `NCCL_ALGO`/`NCCL_PROTO`/channels, or overlap), not link-limited. For this
example TraceLens measured the 48 MB all-gather at **~102 GB/s** bus bandwidth and
the 252 MB all-gather at **~223 GB/s** — but with **inter-rank skew larger than
the comm latency itself**, so this single-node run is skew-bound, not link-bound.
These microbenchmarks benchmark the **fabric**, not this model, so they are a
reference point rather than a per-workload guide.

## Regenerating the screenshots

The Perfetto/TensorBoard figures in these guides are captured **headlessly** on a
compute node (no display) with [`screenshot_perfetto.py`](screenshot_perfetto.py)
and the batch job [`submit_timeline_traces.sbatch`](submit_timeline_traces.sbatch).
The large `.pftrace` / `.proto` / `.pt.trace.json` binaries are **not committed**;
run the batch job to regenerate them and the PNGs under [`figs/`](figs/) in a few
minutes. See each guide's *Capturing the screenshots* section.

## Viewing graphics remotely

Perfetto, TensorBoard, and the TraceLens Excel report are viewed inside a remote
graphical session on AAC6:

- `man aac6_vnc` — TurboVNC desktop (Perfetto, TensorBoard, a spreadsheet viewer)
- `man aac6_novnc` — browser (noVNC) desktop
- `man aac6_x11` — X11 forwarding / SSH tunnel for a single GUI window or port
