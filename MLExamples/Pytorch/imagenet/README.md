# ImageNet DDP: measuring RCCL communication at scale

README.md from `HPCTrainingExamples/MLExamples/Pytorch/imagenet` in the Training Examples repository

The [mnist](../mnist) example is deliberately tiny: the dataset is small and its
"multi-GPU" batch uses `torch.nn.DataParallel`, which is single-process and does
**not** scale well. This example steps up to a **larger workload** (ResNet on
ImageNet-sized 224x224x3 images, 1000 classes) trained with **true
`DistributedDataParallel` (DDP)**, one process per GPU, using the **RCCL**
(ROCm Collective Communication Library) backend.

The key idea: we run with the upstream `--dummy` flag so **no 150 GB ImageNet
download is needed**. `--dummy` feeds `FakeData` generated on the fly, so the
input pipeline is essentially free and each training step is dominated by

1. GPU **compute** (the forward/backward of the CNN), and
2. the **RCCL all-reduce** of gradients across GPUs at the end of each step.

By comparing step time across GPU counts we can isolate and quantify the RCCL
communication cost. This directory adds the tutorial, a scaling-sweep driver,
Slurm batch scripts, and a compute-independent RCCL bandwidth micro-benchmark.

> On ROCm, PyTorch's `nccl` backend is provided by **librccl**. All the
> `NCCL_*` environment variables below are honored by RCCL.

## Contents

| File | Purpose |
|------|---------|
| `ddp_resnet_bench.py` | **Recommended** torchrun DDP ResNet benchmark; synthetic data, `no_sync()` RCCL isolation, `--channels-last`/`--amp` |
| `ddp_bench_sweep.sh` | Scaling driver for `ddp_resnet_bench.py` (pre-warms MIOpen, prints comm%/speedup/efficiency) |
| `submit_ddp_bench.batch` | Slurm job: baseline + optimized (channels_last/AMP) sweeps |
| `warm_miopen.py` | One-shot single-process MIOpen cache warm-up helper |
| `rccl_allreduce_bench.py` | Standalone all-reduce bandwidth benchmark (pure RCCL, no model) |
| `rccl_scaling_sweep.sh` | Sweep driver for the upstream `imagenet/main.py --dummy` (see caveat below) |
| `pytorch_imagenet_ddp_venv.batch` | Slurm job: venv install + full scaling sweep |
| `pytorch_imagenet_ddp_module.batch` | Slurm job: `module load` variant of the sweep |
| `submit_rccl_allreduce.batch` | Slurm job: all-reduce bandwidth vs. GPU count |

This example intentionally **does not copy** `main.py`; it drives the upstream
`pytorch/examples` copy so it always tracks the canonical version.

> [!IMPORTANT]
> **MI300A / AAC6 settings and a caveat about `main.py`.** The PPAC MI300A nodes
> are now booted with `iommu=pt` (**required** — see fix 1 below), so RCCL uses
> direct xGMI P2P and no P2P override is needed. With the default MIOpen tuning,
> though, the naive test can still take many minutes to start. The
> [MI300A settings](#running-on-mi300a-required-settings-and-measured-optimization)
> section documents the required environment and shows that the upstream
> `main.py` `mp.spawn` path can hang on this build (it initializes CUDA before
> spawning). The `ddp_resnet_bench.py` + `ddp_bench_sweep.sh` pair is the
> reliable, measured path and is recommended for the scaling study here.

## Running on MI300A: required settings and measured optimization

This section captures what it actually takes to get the test running well on an
AMD **MI300A** node (measured on AAC6, `PPAC_MI300A_SPX`, ROCm 6.4.3,
PyTorch 2.12), and the optimization result.

### The three fixes that make the test run

1. **RCCL P2P hang** — nodes booted without `iommu=pt` emit
   `NCCL WARN Missing "iommu=pt" ... can lead to ... hang`, and RCCL then hangs
   on the first collective (it can even mark the node down). The cause is that
   intra-node GPU↔GPU peer-to-peer uses IPC handles (`hipIpcGetMemHandle`),
   which need IOMMU passthrough to map the DMA.

   **Real fix (preferred, and now in place on PPAC): boot the node with
   `iommu=pt`.** The PPAC MI300A nodes are now booted with `amd_iommu=on
   iommu=pt` on the kernel command line — this setting **must remain** for the
   collectives to run. Verify on any node with `grep -o 'iommu=pt' /proc/cmdline`.
   With passthrough on, RCCL uses direct xGMI/Infinity Fabric and you set
   **nothing** below.

   **Fallback (node not yet fixed): force collectives off P2P** so they fall
   back to host-staged shared memory. This clears the hang but at a large
   bandwidth cost (no direct xGMI), so use it only until `iommu=pt` is in place:

   ```bash
   export NCCL_P2P_DISABLE=1
   ```

   > `NCCL_P2P_LEVEL=SYS` is **not** the right knob: per the NCCL/RCCL docs `SYS`
   > is the *most* permissive P2P level (it enables P2P across NUMA), not a way to
   > disable it. The host-staging workaround is `NCCL_P2P_DISABLE=1` (equivalently
   > `NCCL_P2P_LEVEL=LOC`). Confirm the transport RCCL actually chose with
   > `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH` (`via P2P`/`xGMI` vs `via SHM`).

2. **MIOpen startup** — the default MIOpen solver search can take **>10 minutes**
   cold for ResNet convolutions, and the site module points the cache at a
   *per-job* directory so every run recompiles. Fix with fast selection + a
   persistent shared cache:

   ```bash
   export MIOPEN_FIND_MODE=FAST
   export MIOPEN_USER_DB_PATH=/tmp/$USER/miopen-shared
   export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
   ```

3. **`mp.spawn` hang** — the upstream `main.py --multiprocessing-distributed`
   calls `torch.accelerator.device_count()` (initializing CUDA) in the parent
   *before* `mp.spawn`, which poisons the child CUDA contexts and hangs RCCL on
   this build. **Use torchrun** instead (independent processes). That is exactly
   what `ddp_resnet_bench.py` does.

`ddp_bench_sweep.sh` sets **no P2P override** (the PPAC nodes have `iommu=pt`, so
RCCL uses direct xGMI), applies fix-2, and pre-warms the MIOpen cache
single-process (so ranks don't contend on the cold cache SQLite db).

> If you ever land on a node **without** `iommu=pt`, export `NCCL_P2P_DISABLE=1`
> before the sweep; the driver scripts pass that through to the ranks. That is a
> host-staged fallback only — the correct fix is to keep `iommu=pt` on the node.

### Recommended workflow

```bash
salloc -p PPAC_MI300A_SPX -N1 --gpus=4 -t 00:20:00
module load rocm openmpi pytorch     # add version pins to sample a combo, e.g. rocm/7.2.3 pytorch/2.12.0

# Baseline weak scaling (per-GPU batch held constant):
GPUS="1 2 4" ARCH=resnet50 BATCH=128 ./ddp_bench_sweep.sh

# Optimized: NHWC memory format + bf16 autocast
GPUS="1 2 4" ARCH=resnet50 BATCH=128 OPTS="--channels-last --amp" ./ddp_bench_sweep.sh

# Add torch.compile (graph capture + kernel fusion) on top:
GPUS="1 2 4" ARCH=resnet50 BATCH=128 OPTS="--channels-last --amp --compile" ./ddp_bench_sweep.sh
```

The optimization levers exposed by `ddp_resnet_bench.py` (pass via `OPTS`):

| Flag | Effect |
|------|--------|
| `--channels-last` | NHWC memory format — matches CDNA conv layout |
| `--amp` | bf16 autocast — the single biggest throughput win |
| `--compile` | `torch.compile` graph capture + kernel fusion; first (warm-up) step pays a one-time compile cost |
| `--migrate` | Stage each host input batch to the GPU with **zero-copy `migrate()`** (MI300A unified memory; needs `HSA_XNACK=1`). See [`../common/README.md`](../common/README.md) |
| `--migrate-method managed\|register` | Zero-copy method: `managed` aliases a `hipMallocManaged` buffer (default); `register` `hipHostRegister`s any pageable tensor (e.g. a DataLoader batch) |
| `--host-copy` | Stage each host input batch with a `.to()` copy — the baseline to compare `--migrate` against |

> **Zero-copy input staging (MI300A).** By default the input batch is
> pre-resident on the GPU. `--host-copy`/`--migrate` instead produce the batch on
> the host each step and move it to the GPU — a `hipMemcpy` copy vs. an aliased
> unified-memory pointer. The raw transfer is ~30× cheaper with `migrate` (see
> [`../common/README.md`](../common/README.md) for the micro-benchmark), but for
> this compute-bound step the reused-buffer copy overlaps compute, so end-to-end
> throughput is within noise. The win shows up for large, per-step-fresh host
> batches and in memory footprint: the copy path keeps a **second device-resident
> copy** of every batch, while `migrate` keeps one (measured **100%** of the
> device duplicate eliminated). Requires `HSA_XNACK=1`.

### Runtime, version, and affinity

See [`../common/PERFORMANCE_NOTES.md`](../common/PERFORMANCE_NOTES.md) for the
cross-cutting, measured results on MI300A: **ROCm/PyTorch version selection**
(ROCm 6.4 vs 7.2.3, TunableOp, and why the pip **wheel vs. site module is a
wash** at matched ROCm), and **NUMA affinity/placement** (negligible for these
GPU-resident runs — enable with `AFFINITY=1 ./ddp_bench_sweep.sh` when you have a
CPU-heavy or host-staged path).

### Measured results (MI300A, resnet50, per-GPU batch 128)

> These numbers were measured earlier with the host-staged P2P workaround (nodes
> **without** `iommu=pt`, busbw ~80 GB/s). The PPAC nodes are now booted with
> `iommu=pt`, so RCCL uses direct xGMI and the all-reduce is several-fold faster —
> `comm%` should drop below what is shown here. Re-run the sweep to refresh these.

Baseline (fp32, NCHW):

```
GPUs   step_s    comm%   img_per_s   speedup   eff
1      0.1297    2.4     987         1.00      100%
2      0.1321    2.9     1938        1.96       98%
4      0.1338    3.0     3826        3.88       97%
```

Optimized (`--channels-last --amp`, bf16):

```
GPUs   step_s    comm%   img_per_s   speedup   eff
1      0.0780    4.0     1641        1.00      100%
2      0.0798    4.8     3208        1.95       98%
4      0.0797    4.1     6424        3.91       98%
```

Takeaways:

- **~1.67x throughput** from AMP bf16 (987 -> 1641 img/s at 1 GPU; 3826 -> 6424 at
  4 GPU). `channels_last` alone was neutral for fp32 convs on this MIOpen build;
  its value shows up together with AMP.
- **Weak-scaling efficiency stays 97-98%** — the RCCL all-reduce of ResNet50's
  ~102 MB of gradients is small and overlaps backward compute.
- **`comm%` rises (2.4% -> ~4%) after optimization**: making compute faster does
  not change the bytes communicated, so RCCL becomes a larger share of a shorter
  step. This is the general rule — the faster your compute, the more communication
  matters, which is why the RCCL study becomes more important as you optimize.

## 0. Get the example and an allocation

The `--dummy` ImageNet trainer lives in the upstream `pytorch/examples` repo:

```bash
git clone --depth=1 https://github.com/pytorch/examples.git ~/pytorch_examples
```

Grab a multi-GPU allocation (8 GPUs shown; adapt the partition to your site):

```bash
salloc --gpus=8 --ntasks=1 --time=01:00:00
```

Set up PyTorch exactly as in the [mnist README](../mnist/README.md) (venv,
container, or module). For a venv:

```bash
python3 -m venv rocm-pytorch && source rocm-pytorch/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip3 install -r ~/pytorch_examples/imagenet/requirements.txt
```

Confirm the GPUs are visible:

```bash
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
amd-smi monitor -g   # or: rocm-smi
```

## 1. A single benchmark step (sanity check)

The example uses `torch.multiprocessing.spawn` internally, so a single command
launches one process per visible GPU. Start with a quick `--dummy` run on all
GPUs of the node:

```bash
cd ~/pytorch_examples/imagenet
python main.py -a resnet50 --dummy \
  --dist-url 'tcp://127.0.0.1:23456' --dist-backend nccl \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  -b 2048 -p 5 --epochs 1
```

You will see progress lines like:

```
Epoch: [0][  5/625]  Time  0.312 (0.318)  Data  0.001 (0.002)  Loss ...  Acc@1 ...
```

The `Time x.xxx (y.yyy)` field is **per-step wall time**: current (running avg).
Read the running average after ~20 steps of warm-up — that is the steady-state
step time we use for scaling. Loss/accuracy are meaningless with `--dummy`;
that is expected. You can stop the run with Ctrl-C once the average settles.

> **`--world-size` here is the number of *nodes*.** With
> `--multiprocessing-distributed` the effective world size becomes
> `nodes x GPUs_per_node`, so `--world-size 1` on an 8-GPU node gives 8 ranks.

## 2. Confirm the RCCL topology

Before measuring, dump how RCCL wired the GPUs (rings/trees, xGMI peers, whether
it fell back to PCIe/SHM). This is the single most useful diagnostic:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
python main.py -a resnet50 --dummy \
  --dist-url 'tcp://127.0.0.1:23456' --dist-backend nccl \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  -b 2048 -p 50 --epochs 1 2>&1 | grep -E 'NCCL|Ring|Channel|Tree' | head -40
```

Look for lines showing `Channel .. via ... [xGMI]` / `P2P` (good, on-fabric) vs.
`via SHM` or `via PCI` (host-staged, slower). On an MI300A node with a healthy
Infinity Fabric mesh you want peer transfers, not SHM.

## 3. Scaling sweep (the RCCL measurement)

`rccl_scaling_sweep.sh` automates runs at 1, 2, 4, 8 GPUs by masking visible
devices (the trainer uses `device_count()` to pick the rank count), extracts the
steady-state step time, and prints throughput, speedup, and efficiency.

Two complementary modes:

**Weak scaling** — hold *per-GPU* batch constant (global batch grows with GPUs).
With a perfect interconnect the step time stays flat, so any increase is RCCL
all-reduce overhead:

```bash
MODE=weak ARCH=resnet50 GPUS="1 2 4 8" PERGPU_BATCH=256 ./rccl_scaling_sweep.sh
```

**Strong scaling** — hold the *global* batch constant. Speedup below N reflects
RCCL cost plus shrinking per-GPU work:

```bash
MODE=strong ARCH=resnet50 GPUS="1 2 4 8" GLOBAL_BATCH=2048 ./rccl_scaling_sweep.sh
```

Example output shape (numbers are illustrative):

```
GPUs   step_time_s    img_per_s    speedup      efficiency
1      0.320          800.0        1.00         100%
2      0.331          1547.0       1.93         97%
4      0.345          2968.0       3.71         93%
8      0.370          5535.0       6.92         86%
```

The drop from 100% is the fraction of time spent in (or stalled behind) the RCCL
gradient all-reduce. `NCCL_DEBUG=INFO ./rccl_scaling_sweep.sh` preserves the full
RCCL logs per run under `rccl_scaling_logs/`.

**Amplify the communication signal** to make RCCL cost obvious:

```bash
# Bigger model => bigger gradient tensors => more bytes all-reduced per step
ARCH=resnet152 MODE=weak GPUS="1 2 4 8" ./rccl_scaling_sweep.sh

# Disable DDP's comp/comm overlap to see the *unhidden* all-reduce cost
NCCL_DEBUG=WARN DDP_OVERLAP=0 ...   # (see step 5 profiler for the clean split)
```

## 4. Pure RCCL bandwidth (compute-independent)

Training throughput mixes compute and communication. To measure RCCL **by
itself**, run the all-reduce micro-benchmark. It launches one rank per GPU with
`torchrun` and reports algorithm and bus bandwidth per message size:

```bash
# 8 ranks on one node
torchrun --standalone --nproc_per_node=8 rccl_allreduce_bench.py

# sweep rank counts by masking GPUs
HIP_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 rccl_allreduce_bench.py
```

Interpretation:

- **algbw** = `size / time` — what the application "sees".
- **busbw** = `algbw x 2(N-1)/N` — normalizes for the ring all-reduce data
  movement so it is comparable across GPU counts and should approach the
  hardware link bandwidth (xGMI/Infinity Fabric).
- **busbw that stays flat** as N grows => the interconnect scales; **busbw that
  falls** => RCCL is the scaling bottleneck for that message size.

The batch script `submit_rccl_allreduce.batch` runs this for 2/4/8 GPUs. If you
have the ROCm **rccl-tests** installed, `all_reduce_perf -b 1M -e 1G -f 2 -g <N>`
is the equivalent lower-level reference.

## 5. Attribute time to RCCL kernels (optional, precise)

To split a real training step into compute vs. communication, wrap the trainer
with a profiler. The `ddp_resnet_bench.py` benchmark exposes this directly:

```bash
# torch.profiler: per-op/kernel + RCCL table, trace per rank under ./torch_prof
torchrun --standalone --nproc_per_node=2 ddp_resnet_bench.py \
  -a resnet50 -b 128 --profile --profile-dir ./torch_prof

# DeepSpeed FlopsProfiler: FLOPs / MACs / params (compute ceiling)
torchrun --standalone --nproc_per_node=1 ddp_resnet_bench.py -a resnet50 -b 64 --flops
```

Quick pointers:

- **torch.profiler**: look for `nccl:all_reduce` / `ncclDevKernel_AllReduce_*`
  entries in the key-averages table; their total is the RCCL time.
- **rocprofv3** (kernel trace): the RCCL collectives show up as `ncclDevKernel_*`
  / all-reduce kernels.
- **rocprof-sys** (timeline): captures host+device activity so you can *see* the
  all-reduce overlapping (or not) with backward compute.

**See [`PROFILING.md`](PROFILING.md)** for the full, MI300A-verified guide
covering torch.profiler, the DeepSpeed FlopsProfiler, TensorBoard, rocprofv3,
rocprof-compute (roofline), rocprofiler-systems (timeline), and multi-node
TAU/HPCToolkit.

## 6. Run it as a batch job

```bash
sbatch pytorch_imagenet_ddp_venv.batch      # venv install + weak & strong sweep
sbatch pytorch_imagenet_ddp_module.batch    # module load variant
sbatch submit_rccl_allreduce.batch          # pure all-reduce bandwidth sweep
```

Tune RCCL from the environment without touching code, e.g.:

```bash
export NCCL_DEBUG=INFO            # topology + collective logging
export NCCL_P2P_DISABLE=1         # force off peer-to-peer (compare vs. on)
export NCCL_MIN_NCHANNELS=8       # more channels for large messages
```

## Useful RCCL environment variables

| Variable | Effect |
|----------|--------|
| `NCCL_DEBUG=INFO` | Print init topology, rings/trees, transport used |
| `NCCL_DEBUG_SUBSYS=INIT,COLL` | Restrict debug output to init + collectives |
| `NCCL_P2P_DISABLE=1` | Disable GPU peer-to-peer (isolate xGMI vs. staged copies) |
| `NCCL_SHM_DISABLE=1` | Disable shared-memory transport |
| `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | Tune channel count / parallelism |
| `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` | Select which GPUs participate |

---

## Alternative examples for studying RCCL scaling

The two families below were evaluated as alternatives. Summary:

| Example | Collective / API | Dataset & model | Best for | Multi-node |
|---------|------------------|-----------------|----------|:----------:|
| **imagenet `--dummy`** (this dir) | DDP all-reduce | FakeData ImageNet, ResNet | First stop: easy, no data, clean weak/strong sweep | yes |
| `distributed/ddp-tutorial-series` | DDP all-reduce | tiny synthetic | Learning the DDP/torchrun launch mechanics | yes |
| `distributed/minGPT-ddp` | DDP all-reduce | char-level text | LLM-shaped gradients (large all-reduce) | yes |
| `distributed/FSDP`, `FSDP2` | all-gather + reduce-scatter | T5 + wikihow | Sharded (ZeRO-3) comm pattern, big models | yes |
| `distributed/tensor_parallelism` | all-reduce/all-gather in-layer | Llama2 block | Intra-layer TP + sequence parallel comm | yes |
| `MLExamples/TinyTransformer` | **none (single GPU)** | synthetic tokens | Single-GPU kernel profiling only | **no** |

### `pytorch_examples/distributed/` — recommended companions

Located at `~/pytorch_examples/distributed/`. These are purpose-built for
distributed training and are the best complements to this example:

- **`ddp-tutorial-series/`** — `single_gpu.py` -> `multigpu.py` ->
  `multigpu_torchrun.py` -> `multinode.py`. Each step adds one piece of the DDP
  stack. Ideal for teaching the **launch mechanics** (process groups, `torchrun`,
  `DistributedSampler`, multi-node rendezvous, the included `slurm/sbatch_run.sh`).
  The model/data are trivial, so it shows *how to scale*, not *how much* — use it
  before this imagenet example. Launch:

  ```bash
  cd ~/pytorch_examples/distributed/ddp-tutorial-series
  torchrun --standalone --nproc_per_node=8 multigpu_torchrun.py 50 10
  ```

- **`minGPT-ddp/`** — a GPT-style transformer trained with DDP. Gradient tensors
  are large and transformer-shaped, so the **all-reduce volume per step is much
  bigger** than ResNet — a more LLM-representative RCCL workload than imagenet.
  Uses the same DDP all-reduce collective, so the measurement methodology in this
  README transfers directly. Config-driven (`gpt2_train_cfg.yaml`) and ships a
  Slurm script. Good "step 2" once imagenet scaling is understood.

  ```bash
  cd ~/pytorch_examples/distributed/minGPT-ddp/mingpt
  torchrun --standalone --nproc_per_node=8 main.py
  ```

- **`FSDP/` and `FSDP2/`** — Fully Sharded Data Parallel (ZeRO-style). The
  collective pattern is **different from DDP**: parameters are `all_gather`-ed
  before use and gradients `reduce_scatter`-ed, so there is more, and more
  frequent, communication. Use these to study the comm pattern of large,
  memory-sharded models (T5 on the wikihow summarization dataset). Higher setup
  cost (real dataset download via `download_dataset.sh`).

- **`tensor_parallelism/`** — tensor/sequence parallelism on a Llama2 block.
  Communication happens **inside** each layer (activation all-reduce/all-gather)
  rather than once per step. Best for studying TP/SP collectives, but the most
  involved to set up.

**Recommendation:** use `ddp-tutorial-series` to learn the launch, this
`imagenet --dummy` example as the primary DDP all-reduce scaling study (no data
needed), and `minGPT-ddp` for an LLM-shaped all-reduce. Move to `FSDP`/
`tensor_parallelism` only when you specifically need sharded or intra-layer
collective patterns.

### `MLExamples/TinyTransformer` — not suitable for RCCL scaling

The [TinyTransformer](../../TinyTransformer) workshop is an excellent
**single-GPU** profiling progression (PyTorch profiler, rocprofv3, rocprof-sys,
rocprof-compute; kernel fusion and Triton optimization of a Tiny-LLaMA). However,
it contains **no distributed code** — no `init_process_group`,
`DistributedDataParallel`, `torchrun`, or collective calls (verified across all
four `version*` implementations). Every run is confined to one GPU, so there is
**no RCCL communication to measure**.

TinyTransformer is the right tool for *per-GPU kernel* optimization and roofline
analysis; it is complementary to, not a substitute for, the DDP examples above.
If you want to study RCCL with a transformer, use `minGPT-ddp` instead. (One could
in principle wrap the Tiny-LLaMA model in DDP to create a scaling exercise, but
that work does not exist in the repo today.)
