# FSDP2 benchmark: rigorous RCCL + memory scaling study

`README_benchmark.md` from `HPCTrainingExamples/MLExamples/Pytorch/FSDP2/benchmarks`
in the Training Examples repository.

This is the in-depth companion to the [quick-start `README.md`](../README.md). Start
there for the simple, manual step-by-step run. This document covers the
scaling-sweep driver, the required MI300A environment and cluster fixes, the
optimization levers (mixed precision, `torch.compile`, zero-copy input staging),
measured results, batch jobs, precise kernel attribution, and how this example
relates to the other distributed examples.

> On ROCm, PyTorch's `nccl` backend is provided by **librccl**. All the
> `NCCL_*` environment variables below are honored by RCCL.
> FSDP2 (`fully_shard`) requires **PyTorch >= 2.5** (ideally a recent release).

## Contents

| File | Purpose |
|------|---------|
| `fsdp2_bench.py` | Shards the upstream transformer with `fully_shard`; reports throughput + peak memory. Flags: `--mixed-precision`, `--compile`, `--profile`, `--flops`, `--rccl-time`, `--migrate`/`--host-copy` |
| `rccl_scaling_sweep.sh` | Runs the benchmark at 2/4/8 GPUs; prints throughput, efficiency, peak memory |
| `pytorch_fsdp2_venv.batch` | Slurm job: venv install + fp32 & bf16 sweeps |
| `pytorch_fsdp2_module.batch` | Slurm job: `module load` variant of the sweep |
| `rccl_scaling_logs/` | Per-run logs from the sweep (RCCL topology if `NCCL_DEBUG=INFO`) |

The benchmark **reuses the upstream** `pytorch/examples/distributed/FSDP2`
transformer (`Transformer`, `ModelArgs`) rather than copying it, so it always
tracks the canonical model. Point it at a checkout with `UPSTREAM=`.

`fsdp2_bench.py` is the polished, all-in-one version of the by-hand patches in
the [quick-start `README.md`](../README.md): the timing/throughput/peak-memory
instrumentation, the `torch.profiler` RCCL timing, `torch.compile`, and mixed
precision are all built in as flags rather than `sed` edits. Use the by-hand
walk-through to *see* the progression; use this script for repeatable sweeps.

## Why two metrics, not just throughput

DDP's communication is a single collective you can toggle off (`no_sync()`), so
its comm cost is easy to isolate. FSDP2's all-gather/reduce-scatter are woven
into forward and backward and cannot simply be switched off. Instead we read the
scaling behavior:

- **peak memory per GPU should fall** as GPU count rises — direct evidence of
  parameter/optimizer-state sharding (the reason to use FSDP2),
- **throughput efficiency below ideal** reflects the growing all-gather +
  reduce-scatter (RCCL) cost.

For exact kernel-level attribution use `--rccl-time` (below) or
`rocprofv3`/`rocprof-sys` (see [`../profiling/PROFILING.md`](../profiling/PROFILING.md)).

## Running on MI300A: required settings and cluster fixes

Measured on AAC6, `PPAC_MI300A_SPX`, ROCm 6.4.3, PyTorch 2.12.

1. **`iommu=pt` (node boot setting, REQUIRED).** The PPAC MI300A nodes must be
   booted with `iommu=pt` (verify: `grep -o 'iommu=pt' /proc/cmdline`) so RCCL
   uses direct xGMI P2P. Without it the all-gather/reduce-scatter hangs. The
   host-staged fallback (slower, use only until the node is fixed) is
   `export NCCL_P2P_DISABLE=1`.

2. **`device_id=` in `init_process_group` (in-code fix, REQUIRED).**
   `fsdp2_bench.py` passes `device_id=device` to `init_process_group`. Without it
   the FSDP2 collectives hang (both ranks spin at ~190% CPU with no progress) on
   this RCCL build. This is separate from — and in addition to — the `iommu=pt`
   node setting.

3. **bf16 hipBLASLt stall (ROCm 7.2.x).** On ROCm 7.2.x the bf16
   (`MIXED_PRECISION=1`) path can **stall for minutes in hipBLASLt**. Run on
   ROCm 6.4.3, or force rocBLAS with `export TORCH_BLAS_PREFER_HIPBLASLT=0`. The
   `hipblaslt/patched` module does **not** fix this and makes no measurable
   difference here — see
   [`../../common/hipblaslt-notes.md`](../../common/hipblaslt-notes.md).

## Recommended workflow

```bash
salloc --gpus=8 --ntasks=1 --time=01:00:00
module load rocm openmpi pytorch     # add version pins to sample a combo, e.g. rocm/7.2.3 pytorch/2.12.0

# Point at the upstream model (only if not in ~/pytorch_examples):
export UPSTREAM=~/pytorch_examples/distributed/FSDP2

# Baseline weak scaling (per-GPU batch held constant, global batch grows):
GPUS="2 4 8" ./rccl_scaling_sweep.sh

# Mixed precision (bf16 params / fp32 reduce) — cuts the all-gather byte volume:
GPUS="2 4 8" MIXED_PRECISION=1 ./rccl_scaling_sweep.sh

# torch.compile the sharded model (graph capture + fusion):
GPUS="2 4 8" OPTS="--compile" ./rccl_scaling_sweep.sh
GPUS="2 4 8" MIXED_PRECISION=1 OPTS="--compile" ./rccl_scaling_sweep.sh

# Push the model larger (where FSDP2 matters most): bigger shards + more comm
GPUS="2 4 8" N_LAYERS=24 DIM=2048 ./rccl_scaling_sweep.sh
```

Illustrative output:

```
GPUs   step_s       tok_per_s      peak_mem_MB    speedup    eff
2      0.0642       255000         7900           1.00       100%
4      0.0551       594000         4300           2.33       116%
8      0.0483       1356000        2310           5.32       133%
```

Two things to read together:

- **peak_mem_MB drops** (7900 -> 2310) as GPUs increase — the sharding win. This
  is what lets FSDP2 train models too large to fit with DDP.
- **throughput scales** but the all-gather/reduce-scatter overhead means the per
  step time does not shrink perfectly linearly. (Super-linear throughput vs. the
  2-GPU base can appear because smaller shards fit better in cache/HBM; compare
  against ideal `N/2` for efficiency.)

### Sweep environment variables

| Variable | Effect (default) |
|----------|------------------|
| `GPUS` | space-separated GPU counts (`"2 4 8"`) |
| `N_LAYERS` / `N_HEADS` / `DIM` / `SEQ` / `BATCH` | model + per-GPU batch (`16`/`16`/`1024`/`512`/`8`) |
| `MIXED_PRECISION=1` | pass `--mixed-precision` (bf16 params, fp32 reduce) |
| `OPTS` | extra `fsdp2_bench.py` flags (e.g. `--compile`) |
| `NCCL_DEBUG=INFO` | log RCCL rings/trees per run under `rccl_scaling_logs/` |
| `AFFINITY=1` | bind each rank to its GPU's local NUMA node (within noise here — see [`../../common/PERFORMANCE_NOTES.md`](../../common/PERFORMANCE_NOTES.md)) |

## Optimization levers (`fsdp2_bench.py` flags)

These flags mirror the by-hand exercises in
[`../README_compute_optimization.md`](../README_compute_optimization.md) and
[`../README_rccl_optimization.md`](../README_rccl_optimization.md) — use those
to *see* where each optimization lives in `example.py`, and the flags here for
repeatable sweeps.

**Precision / compute:**

| Flag | Effect |
|------|--------|
| `--mixed-precision` | bf16 params, fp32 gradient reduce — cuts the all-gather byte volume |
| `--compile` | `torch.compile` the sharded model (graph capture + kernel fusion); the first (warm-up) step pays a one-time compile cost |

**Measurement:**

| Flag | Effect |
|------|--------|
| `--rccl-time` | per-step RCCL kernel device time (all-gather + reduce-scatter) via profiler; reported as `rccl_s` on the RESULT line |
| `--profile [--profile-dir DIR]` | run a few steps under `torch.profiler`, dump a per-rank Kineto/TensorBoard trace |
| `--flops` | rank 0 prints a DeepSpeed FLOPs/MACs/params report of the dense (unsharded) model |

**Input staging (MI300A):**

| Flag | Effect |
|------|--------|
| `--host-copy` | stage each token batch host->GPU with a `.to()` copy — the baseline |
| `--migrate` | stage with **zero-copy `migrate()`** (MI300A unified memory; needs `HSA_XNACK=1`) |
| `--migrate-method managed\|register` | `managed` aliases a `hipMallocManaged` buffer (default); `register` `hipHostRegister`s a pageable buffer |

> **Input staging note.** The token-ID batch is ~2 MB, so `--migrate` vs.
> `--host-copy` does **not** move FSDP2 throughput. The win shows up for large,
> per-step-fresh host batches and in memory footprint (the copy path keeps a
> second device-resident copy; migrate keeps one). See
> [`../../common/README.md`](../../common/README.md) for the 100×–1000× raw-transfer
> micro-benchmark and the memory saving.

> **Runtime / version / affinity.** See
> [`../../common/PERFORMANCE_NOTES.md`](../../common/PERFORMANCE_NOTES.md) for the
> cross-cutting, measured results on MI300A: **ROCm/PyTorch version selection**
> (ROCm 6.4 ~15% faster than 7.2.3 on this sharded transformer; TunableOp on
> 7.2.3 recovers it; pip **wheel vs. site module is a wash** at matched ROCm), and
> **NUMA affinity** (negligible here; enable with `AFFINITY=1`).

## Measured results (MI300A, AAC6 `PPAC_MI300A_SPX`, ROCm 6.4.3 / PyTorch 2.12)

Transformer (16 layers, dim 1024, 16 heads, seq 512, per-GPU batch 8).

> These numbers predate the `iommu=pt` passthrough reboot (host-staged P2P
> workaround), so they should improve — re-run the sweep to refresh them.

Baseline (fp32):

```
GPUs   step_s     tok_per_s   peak_mem_MB   speedup   eff
2      0.1641     49931       7086          1.00      100%
4      0.1566     104615      6285          2.10      105%
```

Optimized (`MIXED_PRECISION=1`, bf16 params / fp32 reduce):

```
GPUs   step_s     tok_per_s   peak_mem_MB   speedup   eff
2      0.0718     114054      4879          1.00      100%
4      0.0806     203167      4081          1.78      89%
```

Takeaways: mixed precision is a **~2.3× throughput** win at 2 GPUs and cuts peak
memory ~30% (7086→4879 MB) by halving the all-gather byte volume. Peak memory
keeps falling as GPUs increase (parameter sharding), the core FSDP2 benefit.
bf16 makes compute cheaper, so the fixed reduce-scatter cost lowers weak-scaling
efficiency from 105% to 89% at 4 GPUs — the RCCL share is now more visible.

## Precise kernel attribution

The benchmark exposes the PyTorch-native profilers directly:

```bash
# torch.profiler: per-op/kernel + all-gather/reduce-scatter table, trace per rank
torchrun --standalone --nproc_per_node=2 fsdp2_bench.py \
  --profile --profile-dir ./torch_prof

# DeepSpeed FlopsProfiler on the dense (unsharded) model: FLOPs / MACs / params
torchrun --standalone --nproc_per_node=1 fsdp2_bench.py --flops
```

For a framework-independent kernel trace:

```bash
rocprofv3 --kernel-trace --stats --truncate-kernels -- \
  torchrun --standalone --nproc_per_node=8 fsdp2_bench.py --warmup 3 --iters 10
```

Look for `ncclDevKernel_AllGather*` and `ncclDevKernel_ReduceScatter*` — these
are the FSDP2 collectives, distinct from DDP's `AllReduce`.

**See [`../profiling/PROFILING.md`](../profiling/PROFILING.md)** for the full guide
covering torch.profiler, the DeepSpeed FlopsProfiler, TensorBoard, rocprofv3,
rocprof-compute (roofline), rocprofiler-systems (timeline), and multi-node
TAU/HPCToolkit.

## Batch jobs

```bash
sbatch pytorch_fsdp2_venv.batch      # venv install + fp32 & bf16 sweeps
sbatch pytorch_fsdp2_module.batch    # module load variant
```

Pin module versions to sample a specific combination, e.g.
`ROCM_VER=6.4.3 PT_VER=2.12.0 sbatch pytorch_fsdp2_module.batch`.

Tune RCCL from the environment without touching code, e.g.:

```bash
export NCCL_DEBUG=INFO            # topology + collective logging
export NCCL_P2P_DISABLE=1         # host-staged fallback (only if no iommu=pt)
export NCCL_MIN_NCHANNELS=8       # more channels for large messages
```

## Run the upstream example (optional)

```bash
cd ~/pytorch_examples/distributed/FSDP2
pip install -r requirements.txt
torchrun --nproc_per_node 8 example.py --mixed-precision
# --explicit-prefetching and --dcp-api are also supported
```

## How this differs from the other examples here

| Example | Collective | What scales | Isolation method |
|---------|-----------|-------------|------------------|
| [imagenet](../../imagenet) | DDP all-reduce | step time (weak/strong) | scaling of step time / `no_sync()` |
| [minGPT-ddp](../../minGPT-ddp) | DDP all-reduce | transformer gradients | `no_sync()` subtraction |
| **FSDP2** (this) | all-gather + reduce-scatter | **memory** and throughput | throughput + peak-memory scaling |

Choose FSDP2 when the model (plus optimizer state) is too large to replicate on
every GPU. For a plain data-parallel all-reduce study, start with `imagenet`;
for an LLM-shaped all-reduce with a direct comm measurement, use `minGPT-ddp`.
