# minGPT-DDP benchmark: rigorous RCCL all-reduce study

`README_benchmark.md` from `HPCTrainingExamples/MLExamples/Pytorch/minGPT-ddp/benchmarks`
in the Training Examples repository.

This is the in-depth companion to the [quick-start `README.md`](../README.md). Start
there for the simple, manual clone-and-patch walk-through. This document covers
the synthetic-data benchmark (`ddp_gpt_bench.py`), its `no_sync()` communication
isolation, the scaling-sweep driver, optimization levers, measured results,
precise kernel attribution, and batch jobs.

> On ROCm, PyTorch's `nccl` backend is provided by **librccl**. All the
> `NCCL_*` environment variables below are honored by RCCL.

## Contents

| File | Purpose |
|------|---------|
| `ddp_gpt_bench.py` | DDP benchmark of the upstream GPT on synthetic tokens; isolates all-reduce cost via `no_sync()`. Flags: `--amp`, `--compile`, `--profile`, `--flops`, `--rccl-time`, `--migrate`/`--host-copy` |
| `rccl_scaling_sweep.sh` | Runs the benchmark at 1/2/4/8 GPUs; prints comm%, throughput, efficiency |
| `pytorch_mingpt_ddp_venv.batch` | Slurm job: venv install + scaling sweep |
| `pytorch_mingpt_ddp_module.batch` | Slurm job: `module load` variant |
| `rccl_scaling_logs/` | Per-run logs from the sweep (RCCL topology if `NCCL_DEBUG=INFO`) |

`ddp_gpt_bench.py` **reuses the upstream** `pytorch/examples/distributed/minGPT-ddp`
model (`GPT`, `GPTConfig`) but drives it with **synthetic tokens** — no dataset
download and no hydra/fsspec/boto3 dependencies — so the RCCL cost is measured
precisely and cheaply. Point it at a checkout with `UPSTREAM=`.

This is the polished, all-in-one version of the by-hand patches in the
[quick-start `README.md`](../README.md): the timing/throughput instrumentation,
the `torch.profiler` RCCL timing (`--rccl-time`/`--profile`), `torch.compile`,
and bf16 autocast are built in as flags rather than `sed` edits. It **adds** the
`no_sync()` direct comm-isolation measurement, which the by-hand walk-through does
not. Use the walk-through to *see* where instrumentation goes; use this script for
repeatable sweeps.

## The key measurement: `no_sync()` isolates the all-reduce

DDP averages gradients across ranks with an **all-reduce** at the end of each
backward pass. PyTorch DDP provides a `no_sync()` context that **skips** that
all-reduce. So for the same model and batch:

```
comm_per_step  ~=  step_time(with all-reduce)  -  step_time(no_sync)
comm_fraction  =   comm_per_step / step_time(with all-reduce)
```

`ddp_gpt_bench.py` times both and reports `comm_pct` directly. This is a cleaner
signal than throughput alone because it separates the RCCL cost from compute — and
it is the measurement the by-hand `torch.profiler` patch in the quick start only
approximates.

## Running on MI300A: required settings

Measured on AAC6, `PPAC_MI300A_SPX`, ROCm 6.4.3, PyTorch 2.12.

1. **`iommu=pt` (node boot setting, REQUIRED).** The PPAC MI300A nodes must be
   booted with `iommu=pt` (verify: `grep -o 'iommu=pt' /proc/cmdline`) so RCCL
   uses direct xGMI P2P. Without it the gradient all-reduce hangs. The host-staged
   fallback (slower, use only until the node is fixed) is `export NCCL_P2P_DISABLE=1`.

2. **bf16 hipBLASLt stall (ROCm 7.2.x).** On ROCm 7.2.x the bf16 (`--amp`)
   transformer path can **stall for minutes in hipBLASLt**. Run bf16 sweeps on
   ROCm 6.4.3, or force rocBLAS with `export TORCH_BLAS_PREFER_HIPBLASLT=0` (then
   bf16 runs and is ~2.3× faster than fp32). The `hipblaslt/patched` module does
   **not** fix this — see [`../../common/hipblaslt-notes.md`](../../common/hipblaslt-notes.md).

## 0. Setup

```bash
git clone --depth=1 https://github.com/pytorch/examples.git ~/pytorch_examples
salloc --gpus=8 --ntasks=1 --time=01:00:00
module load rocm openmpi pytorch     # or a venv/container as in ../../mnist

# Point the benchmark at the upstream model (only if not in ~/pytorch_examples):
export UPSTREAM=~/pytorch_examples/distributed/minGPT-ddp/mingpt
```

The benchmark only needs `torch`; it does **not** need the upstream
`requirements.txt` (hydra/fsspec/boto3) — those are only for the real training job.

## 1. Single run of the benchmark

```bash
torchrun --standalone --nproc_per_node=8 ddp_gpt_bench.py
```

Output (rank 0):

```
# upstream model: /.../minGPT-ddp/mingpt
# world_size=8  params=124.4M  grad_allreduce=498MB/step  per_gpu_batch=8 block=512
RESULT world_size=8 step_sync_s=0.0721 step_nosync_s=0.0605 comm_s=0.0116 comm_pct=16.1 tokens_per_s=454321
```

`comm_pct=16.1` means ~16% of each step is the RCCL gradient all-reduce at this
scale/model size. `grad_allreduce=498MB/step` is how many bytes each rank
contributes to the collective.

## 2. Confirm the RCCL topology

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
  torchrun --standalone --nproc_per_node=8 ddp_gpt_bench.py 2>&1 \
  | grep -E 'NCCL|Ring|Channel|Tree' | head -40
```

Prefer `via ... [xGMI]` / `P2P` (on-fabric) over `via SHM`/`via PCI`.

## 3. Scaling sweep (the RCCL study)

```bash
GPUS="1 2 4 8" ./rccl_scaling_sweep.sh
```

Illustrative output:

```
GPUs   step_s       nosync_s     comm%      tok_per_s      speedup    eff
1      0.0602       0.0602       0.0        108900         1.00       100%
2      0.0631       0.0605       4.1        207700         1.91       95%
4      0.0662       0.0607       8.3        395600         3.63       91%
8      0.0721       0.0605       16.1       454300         6.68       84%
```

At 1 GPU there are no peers, so `comm%`=0 (baseline). As GPUs increase, `comm%`
rises and efficiency falls — that gap **is** the RCCL cost.

**Amplify the communication signal** (bigger model => bigger gradients):

```bash
GPUS="1 2 4 8" N_LAYER=24 N_EMBD=1024 N_HEAD=16 ./rccl_scaling_sweep.sh
```

### Sweep environment variables

| Variable | Effect (default) |
|----------|------------------|
| `GPUS` | space-separated GPU counts (`"1 2 4 8"`) |
| `N_LAYER` / `N_HEAD` / `N_EMBD` / `BLOCK` / `BATCH` | model + per-GPU batch (`12`/`12`/`768`/`512`/`8`) |
| `OPTS` | extra `ddp_gpt_bench.py` flags (e.g. `--amp --compile`) |
| `NCCL_DEBUG=INFO` | log RCCL rings/trees per run under `rccl_scaling_logs/` |
| `AFFINITY=1` | bind each rank to its GPU's local NUMA node (within noise here — see [`../../common/PERFORMANCE_NOTES.md`](../../common/PERFORMANCE_NOTES.md)) |

## Optimization levers (`ddp_gpt_bench.py` flags)

These flags mirror the by-hand exercises in
[`../README_compute_optimization.md`](../README_compute_optimization.md) and
[`../README_rccl_optimization.md`](../README_rccl_optimization.md) — use those to
*see* where each optimization lives in the upstream code, and the flags here for
repeatable sweeps.

**Compute:**

| Flag | Effect |
|------|--------|
| `--amp` | bf16 autocast forward — the biggest single lever for this GEMM-bound transformer |
| `--compile` | `torch.compile` graph capture + kernel fusion; first (warm-up) step pays a one-time compile cost (both the sync and `no_sync` graphs are warmed before timing) |

**Measurement:**

| Flag | Effect |
|------|--------|
| `--rccl-time` | per-step RCCL kernel device time via profiler; reported as `rccl_s` on the RESULT line |
| `--profile [--profile-dir DIR]` | run a few steps under `torch.profiler`, dump a per-rank Kineto/TensorBoard trace |
| `--flops` | rank 0 prints a DeepSpeed FLOPs/MACs/params report |

**Input staging (MI300A):**

| Flag | Effect |
|------|--------|
| `--host-copy` | stage each token batch host->GPU with a `.to()` copy — the baseline |
| `--migrate` | stage with **zero-copy `migrate()`** (MI300A unified memory; needs `HSA_XNACK=1`) |
| `--migrate-method managed\|register` | `managed` aliases a `hipMallocManaged` buffer (default); `register` `hipHostRegister`s a pageable buffer |

> **Input staging note.** minGPT batches are token IDs (~2 MB), far too small for
> `--migrate` vs. `--host-copy` to move end-to-end throughput — it exercises the
> API for completeness. The raw 100×–1000× transfer win and the memory saving
> (the device-resident duplicate eliminated, measured 100%) are in
> [`../../common/README.md`](../../common/README.md).

> **Runtime / version / affinity.** See
> [`../../common/PERFORMANCE_NOTES.md`](../../common/PERFORMANCE_NOTES.md) for the
> cross-cutting, measured MI300A results: **ROCm/PyTorch version selection**
> (ROCm 6.4 ~30% faster than 7.2.3 on this GEMM-bound transformer; TunableOp on
> 7.2.3 recovers it; pip **wheel vs. site module is a wash** at matched ROCm), and
> **NUMA affinity** (negligible here; enable with `AFFINITY=1`).

## Measured results (MI300A, AAC6 `PPAC_MI300A_SPX`, ROCm 6.4.3 / PyTorch 2.12)

GPT2-small (12 layers, 768 embd, 124M params, block 512, per-GPU batch 8).

> These numbers predate the `iommu=pt` passthrough reboot (host-staged P2P
> workaround), so `comm%` should now be lower — re-run the sweep to refresh them.

Baseline (fp32):

```
GPUs   step_s     nosync_s   comm%   tok_per_s   speedup   eff
1      0.0993     0.0959     3.4     41266       1.00      100%
2      0.1061     0.0969     8.7     77196       1.87      94%
4      0.1039     0.0977     6.0     157722      3.82      96%
```

Optimized (`--amp`, bf16 autocast):

```
GPUs   step_s     nosync_s   comm%   tok_per_s   speedup   eff
1      0.0551     0.0529     3.9     74380       1.00      100%
2      0.0616     0.0529     14.0    133093      1.79      90%
4      0.0580     0.0530     8.7     282279      3.80      95%
```

Takeaways: bf16 autocast gives ~**1.8× throughput** at every GPU count; DDP weak
scaling stays 90–96% to 4 GPUs; comm% roughly **doubles** under AMP (14% at 2
GPUs) because the 498 MB all-reduce is unchanged while compute got cheaper.

## 4. Precise kernel attribution

The benchmark exposes the PyTorch-native profilers directly:

```bash
# torch.profiler: per-op/kernel + RCCL all_reduce table, trace per rank
torchrun --standalone --nproc_per_node=2 ddp_gpt_bench.py \
  --profile --profile-dir ./torch_prof

# DeepSpeed FlopsProfiler: FLOPs / MACs / params (compute ceiling)
torchrun --standalone --nproc_per_node=1 ddp_gpt_bench.py --flops
```

For a framework-independent kernel trace:

```bash
rocprofv3 --kernel-trace --stats --truncate-kernels -- \
  torchrun --standalone --nproc_per_node=8 ddp_gpt_bench.py --warmup 5 --iters 10
```

RCCL collectives appear as `ncclDevKernel_AllReduce*`; their total confirms the
`no_sync()` estimate. **See [`../profiling/PROFILING.md`](../profiling/PROFILING.md)**
for the full guide (torch.profiler, DeepSpeed FlopsProfiler, TensorBoard,
rocprofv3, rocprof-compute roofline, rocprof-sys timeline, multi-node TAU/HPCToolkit).

## 5. Batch jobs

```bash
sbatch pytorch_mingpt_ddp_venv.batch      # venv install + scaling sweep
sbatch pytorch_mingpt_ddp_module.batch    # module load variant
```

Pin module versions to sample a specific combination, e.g.
`ROCM_VER=6.4.3 PT_VER=2.12.0 sbatch pytorch_mingpt_ddp_module.batch`.

## 6. Run the real training job (optional)

To train the actual character-level GPT on tinyshakespeare (this is what the
[quick-start `README.md`](../README.md) patches by hand):

```bash
cd ~/pytorch_examples/distributed/minGPT-ddp
pip install -r requirements.txt
torchrun --standalone --nproc_per_node=8 mingpt/main.py \
  trainer_config.max_epochs=1 gpt_config.n_layer=8
```

(Config is hydra-driven from `mingpt/gpt2_train_cfg.yaml`; override on the CLI.)

## How this differs from the other examples here

| Example | Collective | Comm isolation method | Signal |
|---------|-----------|-----------------------|--------|
| [imagenet](../../imagenet) | DDP all-reduce | weak/strong scaling of step time | CNN gradients |
| **minGPT-ddp** (this) | DDP all-reduce | **`no_sync()`** direct subtraction | large transformer gradients |
| [FSDP2](../../FSDP2) | all-gather + reduce-scatter | throughput + peak-memory scaling | sharded params/grads |

Use `imagenet` for the easiest DDP scaling intro, this example when you want an
**LLM-shaped all-reduce** and a direct comm-cost measurement, and `FSDP2` when
the model is too large to replicate per GPU and you need sharding.
