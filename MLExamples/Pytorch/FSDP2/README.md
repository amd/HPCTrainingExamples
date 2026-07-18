# FSDP2: sharded-model RCCL communication and memory scaling

README.md from `HPCTrainingExamples/MLExamples/Pytorch/FSDP2` in the Training Examples repository

This example studies **RCCL** communication for **Fully Sharded Data Parallel
(FSDP2)** — the modern `torch.distributed.fsdp.fully_shard` API. Where DDP (see
[imagenet](../imagenet) and [minGPT-ddp](../minGPT-ddp)) keeps a full copy of the
model on every GPU and does one gradient **all-reduce** per step, FSDP2 **shards**
parameters, gradients, and optimizer state across ranks. That changes the
communication pattern:

- **forward / backward**: parameters are **all-gathered** just-in-time for each
  layer, then freed again,
- **backward**: gradients are **reduce-scattered** back to their owning rank.

So FSDP2 trades **more communication** for **less memory per GPU**. This example
measures both sides of that trade: throughput scaling (how well the RCCL
all-gather/reduce-scatter scales) and peak memory per GPU (how much sharding
saves). It reuses the upstream `pytorch/examples/distributed/FSDP2` transformer.

> On ROCm, PyTorch's `nccl` backend is **librccl**; all `NCCL_*` variables apply.
> FSDP2 (`fully_shard`) requires **PyTorch >= 2.5** (ideally a recent release).

## Contents

| File | Purpose |
|------|---------|
| `fsdp2_bench.py` | Shards the upstream transformer with `fully_shard`; reports throughput + peak memory |
| `rccl_scaling_sweep.sh` | Runs the benchmark at 2/4/8 GPUs; prints throughput, efficiency, peak memory |
| `pytorch_fsdp2_venv.batch` | Slurm job: venv install + fp32 & bf16 sweeps |
| `pytorch_fsdp2_module.batch` | Slurm job: `module load` variant |
| `PROFILING.md` | Full profiling guide: torch.profiler, DeepSpeed FLOPs, TensorBoard, rocprofv3, rocprof-compute, rocprof-sys |

## Why two metrics, not just throughput

DDP's communication is a single collective you can toggle off (`no_sync()`), so
its comm cost is easy to isolate. FSDP2's all-gather/reduce-scatter are woven
into forward and backward and cannot simply be switched off. Instead we read the
scaling behavior:

- **peak memory per GPU should fall** as GPU count rises — direct evidence of
  parameter/optimizer-state sharding (the reason to use FSDP2),
- **throughput efficiency below ideal** reflects the growing all-gather +
  reduce-scatter (RCCL) cost.

For exact kernel-level attribution, use `rocprofv3`/`rocprof-sys` (step 4).

## 0. Setup

```bash
git clone --depth=1 https://github.com/pytorch/examples.git ~/pytorch_examples
salloc --gpus=8 --ntasks=1 --time=01:00:00
```

Set up PyTorch as in the [mnist README](../mnist/README.md), but ensure it is
**>= 2.5**. Verify FSDP2 is importable:

```bash
python3 -c 'from torch.distributed.fsdp import fully_shard; print("FSDP2 OK")'
```

Point the benchmark at the upstream model (only if not in `~/pytorch_examples`):

```bash
export UPSTREAM=~/pytorch_examples/distributed/FSDP2
```

## 1. Single run

FSDP2 needs at least 2 GPUs:

```bash
torchrun --standalone --nproc_per_node=8 fsdp2_bench.py
```

Output (rank 0):

```
# upstream model: /.../distributed/FSDP2
# world_size=8 layers=16 dim=1024 seq=512 per_gpu_batch=8 precision=fp32
RESULT world_size=8 step_s=0.0483 tokens_per_s=678000 peak_mem_mb=2310 local_shard_params=25.9M
```

`peak_mem_mb` is per-GPU peak; `local_shard_params` is how many parameters each
rank actually holds (the full model divided across ranks).

## 2. Confirm the RCCL topology

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
  torchrun --standalone --nproc_per_node=8 fsdp2_bench.py 2>&1 \
  | grep -E 'NCCL|Ring|Channel|Tree' | head -40
```

## 3. Scaling sweep (the RCCL + memory study)

```bash
GPUS="2 4 8" ./rccl_scaling_sweep.sh
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

**Push the model larger** (where FSDP2 matters most):

```bash
GPUS="2 4 8" N_LAYERS=24 DIM=2048 ./rccl_scaling_sweep.sh    # bigger shards + more comm
GPUS="2 4 8" MIXED_PRECISION=1 ./rccl_scaling_sweep.sh        # bf16 params, fp32 reduce
GPUS="2 4 8" OPTS="--compile" ./rccl_scaling_sweep.sh         # torch.compile the sharded model
GPUS="2 4 8" MIXED_PRECISION=1 OPTS="--compile" ./rccl_scaling_sweep.sh
```

Mixed precision reduces the all-gather byte volume (bf16 params) while keeping
the gradient reduce-scatter in fp32 — a common way to cut FSDP communication.
`--compile` wraps the sharded model in `torch.compile` (graph capture + kernel
fusion); the first (warm-up) step pays a one-time compilation cost.

`--migrate` (with `--host-copy` as the copy baseline) stages each token batch
from the host to the GPU via **zero-copy unified-memory aliasing** rather than a
`.to()` copy; `--migrate-method managed|register` picks the mechanism. Requires
`HSA_XNACK=1`. As with minGPT, the token-ID batch is ~2 MB so this does not move
FSDP2 throughput — see [`../common/README.md`](../common/README.md) for the
100×–1000× raw-transfer micro-benchmark and the memory saving (100% of the
device-resident duplicate eliminated), which is where it pays off.

## Measured results (MI300A, AAC6 `PPAC_MI300A_SPX`, ROCm 6.4.3 / PyTorch 2.12)

Transformer (16 layers, dim 1024, 16 heads, seq 512, per-GPU batch 8).

> **Cluster requirement:** the PPAC MI300A nodes must be booted with `iommu=pt`
> (verify: `grep -o 'iommu=pt' /proc/cmdline`) so RCCL uses direct xGMI P2P — no
> `NCCL_P2P_LEVEL`/`NCCL_P2P_DISABLE` override is needed. Without it the
> all-gather/reduce-scatter hangs; the host-staged fallback is
> `NCCL_P2P_DISABLE=1`. The numbers below predate the passthrough reboot, so they
> should improve — re-run the sweep to refresh them.

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

> **Cluster fix (required):** `fsdp2_bench.py` passes `device_id=device` to
> `init_process_group`. Without it the FSDP2 all-gather/reduce-scatter
> collectives hang (both ranks spin at ~190% CPU with no progress) on this RCCL
> build. This is separate from — and in addition to — the required `iommu=pt`
> node setting (host-staged fallback `NCCL_P2P_DISABLE=1` if it is ever absent).

## 4. Precise kernel attribution (optional)

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
are the FSDP2 collectives, distinct from DDP's `AllReduce`. `rocprof-sys` shows
them on a timeline so you can see the all-gather prefetch overlapping compute.

**See [`PROFILING.md`](PROFILING.md)** for the full guide covering torch.profiler,
the DeepSpeed FlopsProfiler, TensorBoard, rocprofv3, rocprof-compute (roofline),
rocprofiler-systems (timeline), and multi-node TAU/HPCToolkit.

## 5. Run the upstream example (optional)

```bash
cd ~/pytorch_examples/distributed/FSDP2
pip install -r requirements.txt
torchrun --nproc_per_node 8 example.py --mixed-precision
# --explicit-prefetching and --dcp-api are also supported
```

## 6. Batch jobs

```bash
sbatch pytorch_fsdp2_venv.batch
sbatch pytorch_fsdp2_module.batch
```

## How this differs from the other examples here

| Example | Collective | What scales | Isolation method |
|---------|-----------|-------------|------------------|
| [imagenet](../imagenet) | DDP all-reduce | step time (weak/strong) | scaling of step time |
| [minGPT-ddp](../minGPT-ddp) | DDP all-reduce | transformer gradients | `no_sync()` subtraction |
| **FSDP2** (this) | all-gather + reduce-scatter | **memory** and throughput | throughput + peak-memory scaling |

Choose FSDP2 when the model (plus optimizer state) is too large to replicate on
every GPU. For a plain data-parallel all-reduce study, start with `imagenet`;
for an LLM-shaped all-reduce with a direct comm measurement, use `minGPT-ddp`.
