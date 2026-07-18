# Profiling the ImageNet DDP example: compute and RCCL communication

The benchmark (`ddp_resnet_bench.py`) already prints a coarse **compute vs.
communication** split from CUDA-event timers — it differences a normal step
against a `no_sync()` step so `comm_s`/`comm_pct` is the gradient all-reduce cost
(see [README §3](README.md) and `RESULT ... comm_pct=`). This guide shows how
to go deeper and attribute time to individual **kernels** (conv/GEMM/batchnorm),
the **RCCL collectives** (`ncclDevKernel_*` / `nccl:all_reduce`), and **hardware
limits** (HBM bandwidth, FLOP/s).

The per-tool steps live in the **shared per-profiler guides** under
[`../common/profilers/`](../common/profilers/README.md) — each page carries the
exact `torchrun` line for this example. Read the ground rules below first.

## Which profiler for which job

| Profiler | Load / install | Scope | Measures here |
|----------|----------------|-------|---------------|
| [**torch.profiler** (Kineto)](../common/profilers/torch-profiler.md) | in `pytorch/2.12.0` | host + GPU | Per-op/kernel table; RCCL `all_reduce` vs. conv/bn; Chrome/Perfetto trace; memory |
| [**DeepSpeed FlopsProfiler**](../common/profilers/deepspeed-flops.md) | in `pytorch/2.12.0` | model (1 GPU) | FLOPs / MACs / params / per-module latency — the compute ceiling |
| [**TensorBoard**](../common/profilers/tensorboard.md) | `pip install` (venv) | host + GPU | GUI for the torch.profiler trace: timeline, kernel view, comm/compute overlap |
| [**rocprofv3**](../common/profilers/rocprofv3.md) | `module load rocm` | GPU | Per-kernel time (MIOpen conv, GEMM) + **RCCL** kernel/API trace |
| [**rocprof-compute**](../common/profilers/rocprof-compute.md) | `module load rocm` | GPU | Roofline: achieved HBM BW / FLOP/s vs. peak for the conv kernels |
| [**rocprofiler-systems**](../common/profilers/rocprofiler-systems.md) | `module load rocprofiler-systems` | GPU + host | Perfetto **timeline**: see all-reduce overlap (or not) with backward |
| [**Score-P**](../common/profilers/scorep.md) | `module load scorep` + venv | Python regions | Per-rank training-loop regions → Cube/OTF2 |
| [**TAU / HPCToolkit**](../../../MPI-examples/cg-solver-example/docs/profilers/README.md) | `module load tau` / `hpctoolkit` | MPI + GPU | Whole-application, **multi-node** MPI + GPU (CG guide) |

`torch.profiler` is the first stop for an ML workload: it speaks the framework's
language (ops, `nccl:all_reduce`, module names) and needs no extra tooling. Drop
to `rocprofv3`/`rocprof-compute` when you need kernel-level or roofline detail,
and to `rocprofiler-systems` when you need to *see* the overlap on a timeline.

> **Verified on PPAC / MI300A (ROCm 6.4.3, `pytorch/2.12.0`, 2×MI300A, resnet50).**
> `torch.profiler` and the DeepSpeed `FlopsProfiler` were run against
> `ddp_resnet_bench.py` via the `--profile` and `--flops` flags added for this
> purpose. See the shared pages for the verified output.

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

## Quick reference

| Goal | Guide | Command |
|------|-------|---------|
| Op/kernel + RCCL table, trace | [torch.profiler](../common/profilers/torch-profiler.md) | `torchrun ... ddp_resnet_bench.py --profile --profile-dir ./torch_prof` |
| Model FLOPs / MACs / params | [DeepSpeed](../common/profilers/deepspeed-flops.md) | `torchrun ... ddp_resnet_bench.py --flops` |
| TensorBoard GUI | [TensorBoard](../common/profilers/tensorboard.md) | `tensorboard --logdir ./torch_prof` |
| Kernel / RCCL trace | [rocprofv3](../common/profilers/rocprofv3.md) | `rocprofv3 --kernel-trace --stats -- torchrun ...` |
| Roofline | [rocprof-compute](../common/profilers/rocprof-compute.md) | `rocprof-compute profile -n r -- torchrun ...` |
| Overlap timeline | [rocprofiler-systems](../common/profilers/rocprofiler-systems.md) | `rocprof-sys-run -- torchrun ...` |
| Per-rank Python regions | [Score-P](../common/profilers/scorep.md) | `NPROC=2 ../common/scorep_launch.sh ddp_resnet_bench.py ...` |
