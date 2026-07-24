# Profiling the FSDP2 example: compute and sharded RCCL communication

The benchmark (`../benchmarks/fsdp2_bench.py`) already reports **throughput** and
**peak memory per GPU** across GPU counts (see the [benchmark
guide](../benchmarks/README_benchmark.md)). Unlike DDP, FSDP2's
communication (`all_gather` of parameters + `reduce_scatter` of gradients) is
woven into forward/backward and cannot be toggled off with `no_sync()`, so the
coarse signal is the scaling behavior. This guide shows how to attribute time
to individual **kernels** (attention / GEMM / layernorm), the **FSDP2
collectives** (`ncclDevKernel_AllGather_*`, `ncclDevKernel_ReduceScatter_*`), and
**hardware limits** (HBM bandwidth, FLOP/s).

The two sharded collectives are what makes FSDP2 different from DDP's single
`all_reduce` — this guide is about *seeing* them.

The per-tool steps live in the **shared per-profiler guides** under
[`../../common/profilers/`](../../common/profilers/README.md) — each page carries the
exact `torchrun` line for this example. Read the ground rules below first.

## Which profiler for which job

| Profiler | Load / install | Scope | Measures here |
|----------|----------------|-------|---------------|
| [**torch.profiler** (Kineto)](../../common/profilers/torch-profiler.md) | in `pytorch/2.12.0` | host + GPU | Per-op/kernel table; `all_gather`/`reduce_scatter` vs. GEMM; Chrome/Perfetto trace; memory |
| [**DeepSpeed FlopsProfiler**](../../common/profilers/deepspeed-flops.md) | in `pytorch/2.12.0` | dense model (1 GPU) | FLOPs / MACs / params of the *unsharded* model — the compute ceiling |
| [**TensorBoard**](../../common/profilers/tensorboard.md) | `pip install` (venv) | host + GPU | GUI for the torch.profiler trace: timeline, kernel view, comm/compute overlap |
| [**rocprofv3**](../../common/profilers/rocprofv3.md) | `module load rocm` | GPU | Per-kernel time (GEMM/attention) + **RCCL** all-gather / reduce-scatter trace |
| [**rocprof-compute**](../../common/profilers/rocprof-compute.md) | `module load rocm` | GPU | Roofline: achieved HBM BW / FLOP/s vs. peak for the GEMM kernels |
| [**rocprofiler-systems**](../../common/profilers/rocprofiler-systems.md) | `module load rocprofiler-systems` | GPU + host | Perfetto **timeline**: see the all-gather *prefetch* overlap with compute |
| [**Score-P**](../../common/profilers/scorep.md) | `module load scorep` + venv | Python regions | Per-rank training-loop regions → Cube/OTF2 |
| [**TAU / HPCToolkit**](../../../../MPI-examples/cg-solver-example/docs/profilers/README.md) | `module load tau` / `hpctoolkit` | MPI + GPU | Whole-application, **multi-node** MPI + GPU (CG guide) |

`torch.profiler` is the first stop for an ML workload: it speaks the framework's
language (ops, `nccl:all_gather`/`nccl:reduce_scatter`, module names) and needs no
extra tooling. Drop to `rocprofv3`/`rocprof-compute` for kernel-level or roofline
detail, and to `rocprofiler-systems` when you need to *see* the prefetch overlap.

> **Verified on PPAC / MI300A** with the site module stack
> `module load rocm/7.2.3 openmpi pytorch/2.12.0` (torch 2.12.0a0, hip 7.2). The
> `--profile` table showed both `nccl:_all_gather_base` and
> `nccl:_reduce_scatter_base` (plus labeled `FSDP::all_gather` rows) and `--flops`
> produced the `FLOPS_PROFILE` line; the module's ROCm-native DeepSpeed imported
> with no `CUDA_HOME` workaround. **Prefer the site modules over pip wheels**: the
> `openmpi` module is GPU-Aware (UCX/UCC/xpmem) and the `pytorch` module is built
> against the matching ROCm. `tensorboard`/`torch-tb-profiler` are **not** in the
> module — install them in a venv (see the [TensorBoard guide](../../common/profilers/tensorboard.md)).

## Ground rules (or your numbers are noise)

Everything in the [quick-start README](../README.md) and the [benchmark
guide](../benchmarks/README_benchmark.md) still applies under a profiler — more
so, because profiling adds overhead:

- Profile **inside an affinity-bound, `--exclusive` allocation** with **≥2 GPUs**
  (FSDP2 shards across ranks; 1 GPU is not a sharded run).
- Nodes must be booted with **`iommu=pt`** so RCCL uses direct xGMI P2P
  (`grep -o iommu=pt /proc/cmdline`). The benchmark also passes `device_id=` to
  `init_process_group` (required — see the [benchmark guide "Cluster
  fix"](../benchmarks/README_benchmark.md)).
- Profile a **fixed, short window** (the `--profile` path uses a
  wait=1/warmup=3/active=6 schedule).
- Keep the **per-rank** traces separate (the flag writes `rank0/`, `rank1/`, …).

## Quick reference

| Goal | Guide | Command |
|------|-------|---------|
| Op/kernel + RCCL table, trace | [torch.profiler](../../common/profilers/torch-profiler.md) | `torchrun ... ../benchmarks/fsdp2_bench.py --profile --profile-dir ./torch_prof` |
| Model FLOPs / MACs / params | [DeepSpeed](../../common/profilers/deepspeed-flops.md) | `torchrun ... ../benchmarks/fsdp2_bench.py --flops` |
| TensorBoard GUI | [TensorBoard](../../common/profilers/tensorboard.md) | `tensorboard --logdir ./torch_prof` |
| Kernel / RCCL trace | [rocprofv3](../../common/profilers/rocprofv3.md) | `rocprofv3 --kernel-trace --stats -- torchrun ...` |
| Roofline | [rocprof-compute](../../common/profilers/rocprof-compute.md) | `rocprof-compute profile -n r -- torchrun ...` |
| Overlap timeline | [rocprofiler-systems](../../common/profilers/rocprofiler-systems.md) | `rocprof-sys-run -- torchrun ...` |
| Per-rank Python regions | [Score-P](../../common/profilers/scorep.md) | `NPROC=2 ../../common/scorep_launch.sh ../benchmarks/fsdp2_bench.py ...` |
