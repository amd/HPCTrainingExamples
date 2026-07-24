# Profiling the minGPT-DDP example: compute and RCCL communication

The benchmark (`../benchmarks/ddp_gpt_bench.py`) already prints a coarse **compute
vs. communication** split from CUDA-event timers — it differences a normal step
against a `no_sync()` step so `comm_s`/`comm_pct` is the gradient all-reduce cost
(see the [benchmark guide](../benchmarks/README_benchmark.md) and the
`RESULT ... comm_pct=` line). This guide
shows how to go deeper and attribute time to individual **kernels** (attention /
GEMM / layernorm), the **RCCL collectives** (`ncclDevKernel_AllReduce_*` /
`nccl:all_reduce`), and **hardware limits** (HBM bandwidth, FLOP/s).

The transformer's gradients are large (≈498 MB/step for the 124M-param GPT2), so
the all-reduce is a **bigger share of each step** than for the ResNet in the
[imagenet](../imagenet) example — this is the more LLM-representative RCCL study.

The per-tool steps live in the **shared per-profiler guides** under
[`../../common/profilers/`](../../common/profilers/README.md) — each page carries the
exact `torchrun` line for this example. Read the ground rules below first.

## Which profiler for which job

| Profiler | Load / install | Scope | Measures here |
|----------|----------------|-------|---------------|
| [**torch.profiler** (Kineto)](../../common/profilers/torch-profiler.md) | in `pytorch/2.12.0` | host + GPU | Per-op/kernel table; RCCL `all_reduce` vs. attention/GEMM; Chrome/Perfetto trace; memory |
| [**DeepSpeed FlopsProfiler**](../../common/profilers/deepspeed-flops.md) | in `pytorch/2.12.0` | model (1 GPU) | FLOPs / MACs / params — the compute ceiling |
| [**TensorBoard**](../../common/profilers/tensorboard.md) | `pip install` (venv) | host + GPU | GUI for the torch.profiler trace: timeline, kernel view, comm/compute overlap |
| [**rocprofv3**](../../common/profilers/rocprofv3.md) | `module load rocm` | GPU | Per-kernel time (GEMM/attention) + **RCCL** kernel/API trace |
| [**rocprof-compute**](../../common/profilers/rocprof-compute.md) | `module load rocm` | GPU | Roofline: achieved HBM BW / FLOP/s vs. peak for the GEMM kernels |
| [**rocprofiler-systems**](../../common/profilers/rocprofiler-systems.md) | `module load rocprofiler-systems` | GPU + host | Perfetto **timeline**: see the all-reduce overlap (or not) with backward |
| [**Score-P**](../../common/profilers/scorep.md) | `module load scorep` + venv | Python regions | Per-rank training-loop regions → Cube/OTF2 |
| [**TAU / HPCToolkit**](../../../../MPI-examples/cg-solver-example/docs/profilers/README.md) | `module load tau` / `hpctoolkit` | MPI + GPU | Whole-application, **multi-node** MPI + GPU (CG guide) |

`torch.profiler` is the first stop for an ML workload: it speaks the framework's
language (ops, `nccl:all_reduce`, module names) and needs no extra tooling. Drop
to `rocprofv3`/`rocprof-compute` for kernel-level or roofline detail, and to
`rocprofiler-systems` when you need to *see* the overlap on a timeline.

> **Verified on PPAC / MI300A** with the site module stack
> `module load rocm/7.2.3 openmpi pytorch/2.12.0` (torch 2.12.0a0, hip 7.2). The
> `--profile` and `--flops` flags added to `ddp_gpt_bench.py` produced the
> `nccl:all_reduce` table row and the `FLOPS_PROFILE` line respectively, and the
> module's ROCm-native DeepSpeed imported with no `CUDA_HOME` workaround.
> **Prefer the site modules over pip wheels**: the `openmpi` module is GPU-Aware
> (UCX/UCC/xpmem) and the `pytorch` module is built against the matching ROCm.
> `tensorboard`/`torch-tb-profiler` are **not** in the module — install them in a
> venv (see the [TensorBoard guide](../../common/profilers/tensorboard.md)).

## Ground rules (or your numbers are noise)

Everything in the [quick-start README](../README.md) and the [benchmark
guide](../benchmarks/README_benchmark.md) still applies under a profiler — more
so, because profiling adds overhead:

- Profile **inside an affinity-bound, `--exclusive` allocation**. Bad GPU/NUMA
  affinity dominates every measurement.
- Nodes must be booted with **`iommu=pt`** so RCCL uses direct xGMI P2P
  (`grep -o iommu=pt /proc/cmdline`). Otherwise the all-reduce is host-staged and
  your "comm" number measures the wrong thing.
- Profile a **fixed, short window** (the `--profile` path uses a
  wait=1/warmup=3/active=6 schedule) — a full run produces an unusable trace.
- Keep the **per-rank** traces separate (the flag writes `rank0/`, `rank1/`, …).

## Quick reference

| Goal | Guide | Command |
|------|-------|---------|
| Op/kernel + RCCL table, trace | [torch.profiler](../../common/profilers/torch-profiler.md) | `torchrun ... ../benchmarks/ddp_gpt_bench.py --profile --profile-dir ./torch_prof` |
| Model FLOPs / MACs / params | [DeepSpeed](../../common/profilers/deepspeed-flops.md) | `torchrun ... ../benchmarks/ddp_gpt_bench.py --flops` |
| TensorBoard GUI | [TensorBoard](../../common/profilers/tensorboard.md) | `tensorboard --logdir ./torch_prof` |
| Kernel / RCCL trace | [rocprofv3](../../common/profilers/rocprofv3.md) | `rocprofv3 --kernel-trace --stats -- torchrun ...` |
| Roofline | [rocprof-compute](../../common/profilers/rocprof-compute.md) | `rocprof-compute profile -n r -- torchrun ...` |
| Overlap timeline | [rocprofiler-systems](../../common/profilers/rocprofiler-systems.md) | `rocprof-sys-run -- torchrun ...` |
| Per-rank Python regions | [Score-P](../../common/profilers/scorep.md) | `NPROC=2 ../../common/scorep_launch.sh ../benchmarks/ddp_gpt_bench.py ...` |
