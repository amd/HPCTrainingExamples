# minGPT-DDP: LLM-shaped RCCL all-reduce scaling

README.md from `HPCTrainingExamples/MLExamples/Pytorch/minGPT-ddp` in the Training Examples repository

This example studies **RCCL** communication for a **transformer (GPT)** trained
with `DistributedDataParallel` (DDP). It complements the
[imagenet](../imagenet) example: same DDP all-reduce collective and same
measurement methodology, but the gradient tensors are **transformer-shaped and
much larger**, so the all-reduce is a bigger fraction of each step — closer to
what you see training real language models.

It builds on the upstream `pytorch/examples/distributed/minGPT-ddp` code:

- The **real training job** (`mingpt/main.py`) trains a character-level GPT on
  tinyshakespeare via `torchrun` + hydra config.
- The **benchmark added here** (`ddp_gpt_bench.py`) reuses the upstream `GPT`
  model but drives it with synthetic tokens so we can measure the RCCL cost
  precisely and cheaply, with no dataset download.

> On ROCm, PyTorch's `nccl` backend is **librccl**; all `NCCL_*` variables apply.

## Contents

| File | Purpose |
|------|---------|
| `ddp_gpt_bench.py` | DDP benchmark of the upstream GPT; isolates all-reduce cost via `no_sync()` |
| `rccl_scaling_sweep.sh` | Runs the benchmark at 1/2/4/8 GPUs; prints comm%, throughput, efficiency |
| `pytorch_mingpt_ddp_venv.batch` | Slurm job: venv install + scaling sweep |
| `pytorch_mingpt_ddp_module.batch` | Slurm job: `module load` variant |

## The key measurement: `no_sync()` isolates the all-reduce

DDP averages gradients across ranks with an **all-reduce** at the end of each
backward pass. PyTorch DDP provides a `no_sync()` context that **skips** that
all-reduce. So for the same model and batch:

```
comm_per_step  ~=  step_time(with all-reduce)  -  step_time(no_sync)
comm_fraction  =   comm_per_step / step_time(with all-reduce)
```

`ddp_gpt_bench.py` times both and reports `comm_pct` directly. This is a cleaner
signal than throughput alone because it separates the RCCL cost from compute.

## 0. Setup

```bash
git clone --depth=1 https://github.com/pytorch/examples.git ~/pytorch_examples
salloc --gpus=8 --ntasks=1 --time=01:00:00
```

Set up PyTorch as in the [mnist README](../mnist/README.md) (venv/container/module).
The benchmark only needs `torch`; the upstream `main.py` additionally needs the
packages in `pytorch_examples/distributed/minGPT-ddp/requirements.txt`
(hydra, fsspec, etc.).

Tell the benchmark where the upstream model lives (only needed if not in the
default `~/pytorch_examples` location):

```bash
export UPSTREAM=~/pytorch_examples/distributed/minGPT-ddp/mingpt
```

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

**Amplify the communication signal:**

```bash
# Bigger model => bigger gradients => larger all-reduce, higher comm%
GPUS="1 2 4 8" N_LAYER=24 N_EMBD=1024 N_HEAD=16 ./rccl_scaling_sweep.sh
```

## 3a. Optimization: bf16 autocast (`--amp`)

The GPT block is GEMM-bound, so bf16 autocast is the single biggest lever. Pass
it through the sweep with `OPTS`:

```bash
GPUS="1 2 4" OPTS="--amp" ./rccl_scaling_sweep.sh
```

Because the fp32 gradients are still all-reduced (498 MB/step for the 124M-param
model), faster bf16 compute makes the fixed RCCL cost a **larger** share of each
step — comm% rises even though wall-clock throughput ~1.8×.

## Measured results (MI300A, AAC6 `PPAC_MI300A_SPX`, ROCm 6.4.3 / PyTorch 2.12)

GPT2-small (12 layers, 768 embd, 124M params, block 512, per-GPU batch 8).

> **Cluster requirement:** the PPAC MI300A nodes must be booted with `iommu=pt`
> (verify: `grep -o 'iommu=pt' /proc/cmdline`) so RCCL uses direct xGMI P2P — no
> `NCCL_P2P_LEVEL`/`NCCL_P2P_DISABLE` override is needed. Without it the gradient
> all-reduce hangs; the host-staged fallback is `NCCL_P2P_DISABLE=1`. The numbers
> below predate the passthrough reboot, so `comm%` should now be lower — re-run
> the sweep to refresh them.

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

## 4. Precise kernel attribution (optional)

```bash
rocprofv3 --kernel-trace --stats --truncate-kernels -- \
  torchrun --standalone --nproc_per_node=8 ddp_gpt_bench.py
```

RCCL collectives appear as `ncclDevKernel_AllReduce*`; their total confirms the
`no_sync()` estimate. `rocprof-sys` gives a timeline showing how much of the
all-reduce overlaps backward compute (DDP overlaps by default).

## 5. Run the real training job (optional)

To train the actual character-level GPT on tinyshakespeare:

```bash
cd ~/pytorch_examples/distributed/minGPT-ddp
pip install -r requirements.txt
torchrun --standalone --nproc_per_node=8 mingpt/main.py \
  trainer_config.max_epochs=1 gpt_config.n_layer=8
```

(Config is hydra-driven from `mingpt/gpt2_train_cfg.yaml`; override on the CLI.)

## 6. Batch jobs

```bash
sbatch pytorch_mingpt_ddp_venv.batch
sbatch pytorch_mingpt_ddp_module.batch
```

## How this differs from the other examples here

| Example | Collective | Comm isolation method | Signal |
|---------|-----------|-----------------------|--------|
| [imagenet](../imagenet) | DDP all-reduce | weak/strong scaling of step time | CNN gradients |
| **minGPT-ddp** (this) | DDP all-reduce | **`no_sync()`** direct subtraction | large transformer gradients |
| [FSDP2](../FSDP2) | all-gather + reduce-scatter | throughput + peak-memory scaling | sharded params/grads |

Use `imagenet` for the easiest DDP scaling intro, this example when you want an
**LLM-shaped all-reduce** and a direct comm-cost measurement, and `FSDP2` when
the model is too large to replicate per GPU and you need sharding.
