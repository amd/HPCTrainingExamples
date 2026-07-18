# Profiling the ImageNet DDP example: compute and RCCL communication

The benchmark (`ddp_resnet_bench.py`) already prints a coarse **compute vs.
communication** split from CUDA-event timers — it differences a normal step
against a `no_sync()` step so `comm_s`/`comm_pct` is the gradient all-reduce cost
(see [README §3](README.md) and `RESULT ... comm_pct=`). This addendum shows how
to go deeper and attribute time to individual **kernels** (conv/GEMM/batchnorm),
the **RCCL collectives** (`ncclDevKernel_*` / `nccl:all_reduce`), and **hardware
limits** (HBM bandwidth, FLOP/s) using the profilers available on the MI300A
nodes plus the PyTorch-native profilers.

## Which profiler for which job

| Profiler | Load / install | Scope | Measures here |
|----------|----------------|-------|---------------|
| **torch.profiler** (Kineto) | in `pytorch/2.12.0` | host + GPU | Per-op/kernel table; RCCL `all_reduce` vs. conv/bn; Chrome/Perfetto trace; memory |
| **DeepSpeed FlopsProfiler** | in `pytorch/2.12.0` | model (1 GPU) | FLOPs / MACs / params / per-module latency — the compute ceiling |
| **TensorBoard** + `torch-tb-profiler` | `pip install` (venv) | host + GPU | GUI for the torch.profiler trace: timeline, kernel view, "GPU utilization", comm/compute overlap |
| **rocprofv3** | `module load rocm` | GPU | Per-kernel time (MIOpen conv, GEMM) + **RCCL** kernel/API trace |
| **rocprof-compute** | `module load rocm` | GPU | Roofline: achieved HBM BW / FLOP/s vs. peak for the conv kernels |
| **rocprofiler-systems** | `module load rocprofiler-systems` | GPU + host | Perfetto **timeline**: see all-reduce overlap (or not) with backward |
| **TAU / HPCToolkit** | `module load tau` / `hpctoolkit` | MPI + GPU | Whole-application, **multi-node** MPI + GPU (see the [CG profiling guide](../../../MPI-examples/cg-solver-example/docs/08-profiling.md)) |

`torch.profiler` is the first stop for an ML workload: it speaks the framework's
language (ops, `nccl:all_reduce`, module names) and needs no extra tooling. Drop
to `rocprofv3`/`rocprof-compute` when you need kernel-level or roofline detail,
and to `rocprofiler-systems` when you need to *see* the overlap on a timeline.

> **Verified on PPAC / MI300A (ROCm 6.4.3, `pytorch/2.12.0`, 2×MI300A, resnet50).**
> `torch.profiler` and the DeepSpeed `FlopsProfiler` were run against
> `ddp_resnet_bench.py` via the `--profile` and `--flops` flags added for this
> purpose. `torch-tb-profiler`/`tensorboard` are **not** in the module — install
> them in a venv (see §C). rocprofv3/rocprof-compute/rocprofiler-systems commands
> follow the same methodology verified in the CG guide.

## Ground rules (or your numbers are noise)

Everything in [README §3–5](README.md) still applies under a profiler — more so,
because profiling adds overhead:

- Profile **inside an affinity-bound, `--exclusive` allocation**. Bad GPU/NUMA
  affinity dominates every measurement.
- Nodes must be booted with **`iommu=pt`** so RCCL uses direct xGMI P2P
  (`grep -o iommu=pt /proc/cmdline`). Otherwise the all-reduce is host-staged and
  your "comm" number measures the wrong thing.
- **Warm up MIOpen first** (`export MIOPEN_FIND_MODE=FAST`, or run `warm_miopen.py`)
  so conv autotuning does not land inside the profiled window.
- Profile a **fixed, short window** (the `--profile` path uses a
  wait=1/warmup=3/active=6 schedule) — a full epoch produces an unusable trace.
- Keep the **per-rank** traces separate (the flag writes `rank0/`, `rank1/`, …).

---

## A. torch.profiler (Kineto) — the primary ML profiler

The `--profile` flag wraps a few steps in `torch.profiler.profile` with the
`CPU` + `CUDA` activity sets. On ROCm the `CUDA` set captures HIP kernels **and**
the RCCL collective kernels, so one table attributes time to compute vs.
communication.

```bash
module load rocm openmpi pytorch
export MIOPEN_FIND_MODE=FAST
# 2 GPUs, dump a trace per rank under ./torch_prof/rank{0,1}
torchrun --standalone --nproc_per_node=2 ddp_resnet_bench.py \
  -a resnet50 -b 128 --profile --profile-dir ./torch_prof
```

Rank 0 prints the key-averages table (sorted by CUDA time). Verified top rows for
resnet50 on MI300A — the workload is convolution-bound:

```
Name                                   Self CUDA   Self CUDA %   # of Calls
DistributedDataParallel.forward        173.652ms      50.81%          6
aten::convolution_backward             151.917ms      44.45%        318
aten::miopen_convolution                68.875ms      20.15%        318
aten::miopen_batch_norm_backward        30.013ms       8.78%        318
...
Self CUDA time total: 341.746ms
```

**Finding the RCCL cost.** Sort by name or search the table for `nccl` /
`AllReduce`; the `ncclDevKernel_AllReduce_*` (and the `nccl:all_reduce` op) rows
are the communication. For resnet50 they are small relative to conv, which is
exactly why `comm_pct` is low and DDP hides most of it behind backward. To make
the RCCL signal larger, profile a bigger model (`-a resnet152`) or the
`minGPT-ddp` example (transformer-shaped, larger gradients).

Useful knobs (edit `profile_steps` in `ddp_resnet_bench.py` if you want them):

- `record_shapes=True` — group kernels by input shape (already on).
- `profile_memory=True` — the `CUDA Mem` / `Self CUDA Mem` columns (already on).
- `with_stack=True` — Python call stacks (needed for the TensorBoard source view;
  adds overhead).
- `sort_by="self_cuda_time_total"` — isolate the heaviest single kernels.

The trace is written as `*.pt.trace.json` (one per rank). View it in:

- `chrome://tracing` (drag-and-drop), or
- [https://ui.perfetto.dev](https://ui.perfetto.dev) (drag-and-drop), or
- TensorBoard (§C).

## B. DeepSpeed FlopsProfiler — the compute ceiling

`--flops` runs the DeepSpeed `get_model_profile` on the model **without** needing
a DeepSpeed engine (it works on any `nn.Module`), reporting FLOPs, MACs, params,
and per-module forward latency. This is the theoretical compute cost to compare
against the measured kernel time from §A / the roofline from §E.

```bash
torchrun --standalone --nproc_per_node=1 ddp_resnet_bench.py \
  -a resnet50 -b 64 --flops
```

Verified output (resnet50, batch 64):

```
fwd MACs per GPU:  261.71 GMACs
FLOPS_PROFILE arch=resnet50 batch=64 flops=525.51 G macs=261.71 GMACs params=25.56 M
```

Interpretation: 525.5 GFLOPs forward for a 64-image batch, 25.56 M params →
`25.56M × 4 B = 102 MB` all-reduced per step in fp32 (the `grad_allreduce=` field
of the normal `RESULT` line). Combine with §A to get achieved TFLOP/s
(`FLOPs / kernel_time`) and with §D to see whether 102 MB all-reduce is
link-bound.

> DeepSpeed's FlopsProfiler is the right "profiler" for this DDP example because
> the model is a plain `nn.Module`. The **full** DeepSpeed profiler (comm logging,
> ZeRO stage timing) only applies if you convert training to `deepspeed.initialize`
> — out of scope here; use `torch.profiler` for the DDP comm breakdown instead.

## C. TensorBoard (torch-tb-profiler plugin)

The `*.pt.trace.json` files from §A load directly into TensorBoard's PyTorch
Profiler plugin, which gives the "Overview", "Operator", "Kernel", "Trace", and
(with `with_stack=True`) source views — including a GPU-utilization estimate and
a step-time breakdown into compute / communication / other.

`tensorboard` and `torch-tb-profiler` are **not** in the `pytorch/2.12.0` module,
so install them into a venv layered on the module:

```bash
module load rocm openmpi pytorch
python -m venv --system-site-packages ~/venvs/tbprof
source ~/venvs/tbprof/bin/activate
pip install tensorboard torch-tb-profiler

# point TensorBoard at the trace dir written by --profile
tensorboard --logdir ./torch_prof --port 6006
# then browse http://<node>:6006  (ssh -L 6006:<node>:6006 for a remote node)
```

`tensorboard_trace_handler` (used by `--profile`) already writes the exact layout
the plugin expects, so `--logdir ./torch_prof` "just works". Use the **Trace**
view to confirm the RCCL all-reduce overlaps the backward pass, and the
**Kernel** view for the same conv-dominated breakdown seen in §A.

## D. rocprofv3 — kernel and RCCL trace

For a framework-independent kernel trace (and to see the RCCL collectives as
device kernels / API calls):

```bash
module load rocm openmpi pytorch
export MIOPEN_FIND_MODE=FAST
# --profile keeps the window short; rocprofv3 traces the whole process, so use a
# small iters/warmup run instead to bound the trace size:
rocprofv3 --kernel-trace --stats --truncate-kernels -- \
  torchrun --standalone --nproc_per_node=2 ddp_resnet_bench.py \
    -a resnet50 -b 128 --iters 10 --warmup 5
```

The stats CSV ranks kernels by total time (MIOpen conv variants dominate). To
trace the collectives explicitly add the RCCL/HIP API traces:

```bash
rocprofv3 --hip-trace --rccl-trace --stats -- \
  torchrun --standalone --nproc_per_node=2 ddp_resnet_bench.py -a resnet50 -b 128 --iters 10
```

Look for `ncclDevKernel_AllReduce_*` in the kernel trace and the RCCL API rows in
the RCCL trace; their total is the communication time, comparable to `comm_s`.

## E. rocprof-compute — roofline of the conv kernels

Prove whether the convolutions are compute- or memory-bound on MI300A:

```bash
module load rocm openmpi pytorch
export MIOPEN_FIND_MODE=FAST
rocprof-compute profile -n resnet_roof -- \
  torchrun --standalone --nproc_per_node=1 ddp_resnet_bench.py \
    -a resnet50 -b 128 --iters 10 --warmup 5
rocprof-compute analyze -p workloads/resnet_roof/MI300A_A1 --gui   # or text tables
```

Compare achieved TFLOP/s to the §B FLOP count and to the MI300A peak; conv on
CDNA benefits from `--channels-last` (NHWC), so profile with and without it to
quantify the win the README already recommends.

## F. rocprofiler-systems — the overlap timeline

To *see* the all-reduce overlapping (or stalling) the backward pass:

```bash
module load rocprofiler-systems
rocprof-sys-run -- \
  torchrun --standalone --nproc_per_node=2 ddp_resnet_bench.py \
    -a resnet50 -b 128 --iters 10 --warmup 5
# open the resulting perfetto-trace.proto in https://ui.perfetto.dev
```

On the timeline the RCCL stream should run *concurrently* with the backward
compute stream — that overlap is exactly what keeps `comm_pct` low. A serialized
all-reduce (comm after compute finishes) points to a bucketing/overlap problem or
a host-staged transport (check `iommu=pt` and `NCCL_DEBUG=INFO`).

## G. Multi-node / MPI whole-application (TAU, HPCToolkit)

The tools above are single-node. For **multi-node** runs launched under MPI, or
to attribute host-side MPI/launch time, use TAU or HPCToolkit exactly as
documented (with verified commands) in the CG example's profiling guide:
[`MPI-examples/cg-solver-example/docs/08-profiling.md`](../../../MPI-examples/cg-solver-example/docs/08-profiling.md)
(§D TAU, §E HPCToolkit). Those sections show per-rank output dirs, `pprof`/`hpcprof`
post-processing, and the module-conflict fixes needed on these nodes.

---

## Quick reference

| Goal | Command |
|------|---------|
| Op/kernel + RCCL table, trace | `torchrun ... ddp_resnet_bench.py --profile --profile-dir ./torch_prof` |
| Model FLOPs / MACs / params | `torchrun ... ddp_resnet_bench.py --flops` |
| TensorBoard GUI | `pip install tensorboard torch-tb-profiler; tensorboard --logdir ./torch_prof` |
| Kernel trace (stats) | `rocprofv3 --kernel-trace --stats -- torchrun ...` |
| RCCL trace | `rocprofv3 --hip-trace --rccl-trace --stats -- torchrun ...` |
| Roofline | `rocprof-compute profile -n r -- torchrun ...` then `rocprof-compute analyze -p workloads/r/MI300A_A1` |
| Overlap timeline | `rocprof-sys-run -- torchrun ...` → Perfetto |
