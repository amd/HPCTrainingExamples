# ImageNet DDP: measuring RCCL communication at scale

README.md from `HPCTrainingExamples/MLExamples/Pytorch/imagenet` in the Training Examples repository

The [mnist](../mnist) example is deliberately tiny: the dataset is small and its
"multi-GPU" batch uses `torch.nn.DataParallel`, which is single-process and does
**not** scale well. This example steps up to a **larger workload** (ResNet on
ImageNet-sized 224x224x3 images, 1000 classes) trained with **true
`DistributedDataParallel` (DDP)**, one process per GPU, using the **RCCL**
(ROCm Collective Communication Library) backend.

The key idea: synthetic (`--dummy`-style) data means **no 150 GB ImageNet
download is needed**. The input pipeline is essentially free, so each training
step is dominated by

1. GPU **compute** (the forward/backward of the CNN), and
2. the **RCCL all-reduce** of gradients across GPUs at the end of each step.

By comparing step time across GPU counts we isolate and quantify the RCCL
communication cost.

> This README is the **quick start**. For the required MI300A settings, the
> scaling-sweep drivers, optimization levers, measured results, profiling, and
> the pure-RCCL bandwidth micro-benchmark, see
> **[`benchmarks/README_benchmark.md`](benchmarks/README_benchmark.md)**.

> On ROCm, PyTorch's `nccl` backend is provided by **librccl**, so all the
> `NCCL_*` environment variables are honored by RCCL.

## 1. Get an allocation and load PyTorch

```bash
salloc -p PPAC_MI300A_SPX -N1 --gpus=4 -t 00:40:00
```

> Set up virtual environment to avod scattering python packages across system
>    and for more repeatability

```bash
uv init imagenet_test
cd imagenet_test
uv venv --system-site-packages
source .venv/bin/activate
```

> Use pre-installed module versions to avoid downloading large wheels.
> uv pip install -r requirements.txt # installs nvidia packages, so skip

```bash
module load rocm openmpi pytorch
```

## 2. Get the examples

```bash
git clone --depth=1 https://github.com/pytorch/examples.git pytorch_examples
cp pytorch_examples/imagenet/* .
```

## 3. Optional modifications to the example source code

> Upstream main.py inits the NCCL process group but never destroys it, so PyTorch
> warns at exit ("`destroy_process_group()` was not called ... can leak resources").
> Register an atexit handler right after `init_process_group` so every worker
> (incl. mp.spawn children) cleans up on a normal exit.

```bash
sed -i '/world_size=args.world_size, rank=args.rank)/a\        import atexit as _ax, torch.distributed as _d; _ax.register(lambda: _d.destroy_process_group() if _d.is_initialized() else None)' main.py
```

> Print per-GPU peak memory once at the end of train() (matches README `peak_mem_mb`)
```bash
sed -i '/^def validate(/i\    torch.cuda.is_available() and getattr(args,"rank",0)<=0 and print(f"PEAK_MEM_MB {torch.cuda.max_memory_allocated()/1e6:.0f}")' main.py
```

> --- Demo instrumentation: total RCCL time + .to vs .migrate staging time ---
> The migrate path (STAGE=migrate) aliases the batch instead of copying it; it
> needs these and HSA_XNACK=1 (exported below).
```bash
export COMMON_DIR="../common"
export HSA_XNACK=1
```

### 1) Start a profiler and set up staging counters at the top of train().
```bash
sed -i '/^    model.train()/a\
    import torch.profiler as _tp\
    _prof = _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA]); _prof.start()\
    _stage_ms = 0.0; _stage_n = 0; _stager = None\
    if os.environ.get("STAGE") == "migrate":\
        import sys; sys.path.insert(0, os.environ["COMMON_DIR"]); from zerocopy import Stager\
        _stager = Stager(device, method="register")' main.py
```

### 2) Keep the demo (and the profiler trace) short.
```bash
sed -i '/^        data_time.update(time.time() - end)/a\
        if i >= 100: break' main.py
```

### 3) Time the host->device staging, but only when STAGE is set (copy vs migrate).
```bash
sed -i '/^        images = images.to(device, non_blocking=True)/c\
        if os.environ.get("STAGE"):\
            _e0 = torch.cuda.Event(enable_timing=True); _e1 = torch.cuda.Event(enable_timing=True)\
            _e0.record()\
            images = _stager.to_device(images) if _stager is not None else images.to(device, non_blocking=True)\
            _e1.record(); torch.cuda.synchronize()\
            _stage_ms += _e0.elapsed_time(_e1); _stage_n += 1\
        else:\
            images = images.to(device, non_blocking=True)' main.py
```

### 4) register-migrate needs pageable (non-pinned) host memory.
```bash
sed -i 's/pin_memory=True/pin_memory=False/g' main.py
```

### 5) Stop the profiler and print total RCCL kernel time (+ staging time if timed).
```bash
sed -i '/^def validate(/i\
    _prof.stop()\
    _rccl_ms = sum(e.self_device_time_total for e in _prof.key_averages() if "nccl" in e.key.lower())/1e3\
    _ws = getattr(args,"world_size","?"); _stg = os.environ.get("STAGE","")\
    getattr(args,"rank",0)<=0 and print(f"RCCL_TOTAL_MS {_rccl_ms:.3f} gpus={_ws}")\
    getattr(args,"rank",0)<=0 and _stage_n and print(f"STAGE_MS_PER_STEP {_stage_ms/_stage_n:.4f} gpus={_ws} stage={_stg}")' main.py
```

Notes:

> MIOpen's default solver search can take **>10 minutes** cold for ResNet
> convolutions. Set fast selection, then warm the cache by running the warmup script
> or the **1-GPU `main.py` case** (the same run the sweep uses) for a few steps:

> The pytorch module already set `MIOPEN_USER_DB_PATH` / `MIOPEN_CUSTOM_CACHE_DIR`
> at a stable per-allocation dir (e.g. /tmp/$USER/miopen-cache/jobs/<jobid>), so
> DON'T override them -- just inherit them. Only ensure fast solver selection:

```bash
export MIOPEN_FIND_MODE=FAST
```

> Suppress some warning noise
```bash
export MIOPEN_LOG_LEVEL=3
export KINETO_LOG_LEVEL=3
```

> `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` are set in the python module to a `/tmp` directory
> Create the directory and set the automatic removal at end of job

```bash
mkdir -p "$MIOPEN_USER_DB_PATH"
```

Confirm the GPUs are visible:

```bash
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

## 3. Warm the MIOpen cache (once per allocation)

> **Warm once per allocation.** A new `salloc`/`sbatch` gets a fresh job ID (and
> thus a fresh empty cache). Warming single-process first matters: inside one
> allocation all N ranks share that one cache dir, so a cold multi-rank run would
> contend on the SQLite db; after warming, ranks just read it (with
> `MIOPEN_FIND_MODE=FAST`).

```bash
HIP_VISIBLE_DEVICES=0  python -c "import torch,torchvision.models as M; \
   d=torch.device('cuda'); \
   n=M.resnet50().to(d); \
   c=torch.nn.CrossEntropyLoss().to(d); \
   x=torch.randn(256,3,224,224,device=d); \
   y=torch.randint(0,1000,(256,),device=d); \
   [c(n(x),y).backward() for _ in range(3)];  \
   torch.cuda.synchronize(); \
   print('warm done')"
```

## 4. Run the scaling sweep (one line per GPU count)

Run the benchmark once per GPU count by changing `HIP_VISIBLE_DEVICES`.

```bash
HIP_VISIBLE_DEVICES=0       python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
        --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 128  -p 20 --epochs 1 2>&1 | tee run_1.log
HIP_VISIBLE_DEVICES=0,1     python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
        --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 256  -p 20 --epochs 1 2>&1 | tee run_2.log
HIP_VISIBLE_DEVICES=0,1,2,3 python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
        --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 512  -p 20 --epochs 1 2>&1 | tee run_4.log
```

## 4. APU programming model (MI300A)

> The MI300A APU has a unified memory and does not need to copy the data, just the pointer. Other GPUS can emulate APU behavior
>   Requires `HSA_XNACK 1` to be set. Set earlier in script
>   `.to` (copy) vs `.migrate` staging comparison (4 GPUs): compare `STAGE_MS_PER_STEP` in final report

- **Host-to-device staging time** — the per-step `images.to(device)` copy is
  wrapped in CUDA events and printed as `STAGE_MS_PER_STEP`, but **only when the
  `STAGE` environment variable is set**, so the plain scaling runs are not
  perturbed. Setting `STAGE=migrate` swaps the `.to()` copy for the zero-copy
  `migrate()` path (MI300A unified memory), so you can compare the two directly:

```bash
# .to() copy vs zero-copy migrate (single GPU) -- compare STAGE_MS_PER_STEP
STAGE=copy    HIP_VISIBLE_DEVICES=0 python main.py -a resnet50 --dummy \
  --dist-url 'tcp://127.0.0.1:23456' --dist-backend nccl \
  --multiprocessing-distributed --world-size 1 --rank 0 -b 128 -p 20 --epochs 1 2>&1 | tee stage_copy.log
STAGE=migrate HIP_VISIBLE_DEVICES=0 python main.py -a resnet50 --dummy \
  --dist-url 'tcp://127.0.0.1:23456' --dist-backend nccl \
  --multiprocessing-distributed --world-size 1 --rank 0 -b 128 -p 20 --epochs 1 2>&1 | tee stage_migrate.log
```

The `.to()` path pays a `hipMemcpy` every step; `migrate` aliases the batch, so
its `STAGE_MS_PER_STEP` should be much smaller. `migrate` requires `HSA_XNACK=1`
and `COMMON_DIR` pointing at [`../common`](../common) (both exported by
`main.py`); if the migrate extension can't build, `Stager` falls back to a copy
and the two numbers will match.


## 5. Measure RCCL time and compare `.to` vs `.migrate` staging

The [`main.py`](main.py) driver runs the sweep above **and** adds two extra
numbers with a handful of small `sed` patches to the (freshly cloned) `main.py`.
The patches are deliberately tiny so they are clear in a hands-on session:

- **Total RCCL time** — a `torch.profiler` is started at the top of `train()` and
  stopped at the end; the on-GPU time of the `nccl*` collective kernels is summed
  and printed as `RCCL_TOTAL_MS`. This is the total RCCL communication time for
  the run (it is ~0 at 1 GPU, since there is no all-reduce, and grows with GPU
  count).

> Get the RCCL total time for each run sorted by the number of GPUs
```bash
echo "=== RCCL total time (per GPU count) ==="
grep -h RCCL_TOTAL_MS run_*.log | sort -t= -k2 -n
```

> Scaling runs are run_<N>.log; staging runs are stage_{copy,migrate}.log, so a
> grep keeps the two reports separate. Lines are self-describing (gpus=N).
```bash
echo "=== Host->device staging: .to (copy) vs .migrate ==="
grep -h STAGE_MS_PER_STEP stage_*.log
```

### 6. Calculating the performance

```bash
echo "=== Calculating the performance =="
./images_per_sec.sh
```

Stock upstream `main.py` prints periodic `Epoch:` progress lines that include the
per-step `Time`. The [`images_per_sec.sh`](images_per_sec.sh) helper parses those
logs (`run_1.log`, `run_2.log`, `run_4.log` written by
[`main.py`](main.py)) and prints one line per GPU count:

```
run_1.log  img/s=968   step=0.1322s  batch=128  peak_mem_mb=...  speedup=1.00x
run_2.log  img/s=1901  step=0.1347s  batch=256  peak_mem_mb=...  speedup=1.96x
run_4.log  img/s=3720  step=0.1376s  batch=512  peak_mem_mb=...  speedup=3.84x
```

- **`img/s`** — global throughput, computed as the total node batch divided by
  the average per-step `Time`. Ideally it grows linearly with GPU count; the gap
  from linear is RCCL cost.
- **`speedup`** — throughput relative to the 1-GPU baseline (`run_1.log`), so it
  reads `1.00x` for `run_1` and shows the weak-scaling efficiency for `run_2`/`run_4`.
- **`step`** — average per-step time. `-b` here is the **per-GPU** batch, so the
  global batch grows with GPU count (weak scaling): flat step time = perfect
  scaling; any growth is RCCL.
- **`peak_mem_mb`** — per-GPU peak allocated memory (`PEAK_MEM_MB`).

> This is the simple, demo-friendly version. For the robust, per-step, per-rank
> instrumentation (with a fair pinned-memory baseline) use
> `ddp_resnet_bench.py`'s `--rccl-time`, `--host-copy`, and `--migrate` flags,
> documented in [`benchmarks/README_benchmark.md`](benchmarks/README_benchmark.md).

## 7. Cleanup

```
deactivate
rm -rf imagenet_test
```

## 8. Run on CPX partitions (`SH5_MI300A_CPX`, `PPAC_MI300A_CPX`)

The sweep above assumes **SPX** mode, where each MI300A APU is one HIP device
(so `PPAC_MI300A_SPX --gpus=4` = 4 devices). When SPX nodes are scarce, the same
study runs on **CPX** partitions, where each of an APU's **6 XCDs** is exposed as
its own HIP device:

| Partition | Physical APUs | HIP devices | What one device is |
|---|---|---|---|
| `SH5_MI300A_CPX` | 1 | 6 | one XCD (~1/6 of an APU) |
| `PPAC_MI300A_CPX` | 4 | 24 | one XCD (~1/6 of an APU) |

Two ready-to-submit batch scripts drive the CPX sweeps:

```bash
sbatch run_imagenet_uv_sh5_cpx.sbatch    # 1 APU, GPU_LIST="1 2 4 6"
sbatch run_imagenet_uv_ppac_cpx.sbatch   # 4 APUs, GPU_LIST="1 2 4 6 12 24"
```

Both are like the SPX driver but with the sweep written as a loop over a
GPU-count list, and two CPX-specific adjustments:

- **Smaller per-GPU batch.** A CPX partition has ~1/6 the compute and memory of a
  full APU, so the SPX per-GPU batch of 128 can OOM or crawl. The scripts default
  to `PERGPU_BATCH=32`; the global batch is `N * PERGPU_BATCH` (weak scaling).
  Both `GPU_LIST` and `PERGPU_BATCH` are overridable at submit time, e.g.
  `sbatch --export=ALL,PERGPU_BATCH=64 run_imagenet_uv_sh5_cpx.sbatch`. Tune it up
  until `peak_mem_mb` approaches the partition's memory limit (which depends on
  whether the node uses shared (NPS1) or split (NPS4) memory — check `rocm-smi`).

- **Extended GPU-count list.** `SH5` goes up to 6 (one XCD → the whole chip);
  `PPAC` continues to 12 and 24.

`images_per_sec.sh` handles either sweep: it derives the GPU count `N` from each
`run_<N>.log` name and reads `PERGPU_BATCH` from the environment, printing the
same `img/s` / `speedup` lines (speedup is relative to the smallest `N`).

**Interpreting CPX results.** The RCCL story differs sharply between the two:

- On **`SH5_MI300A_CPX`** all ranks live on one APU, so the gradient all-reduce
  travels over the on-package Infinity Fabric. It is extremely fast, so
  `RCCL_TOTAL_MS` / `comm` cost stays near zero even at 6 GPUs — communication
  looks almost free.
- On **`PPAC_MI300A_CPX`** up to 6 ranks stay intra-APU (cheap), but at `N=12`
  and `N=24` the collective crosses physical APUs (socket-to-socket links), so
  RCCL cost should visibly rise. This is the CPX sweep that best reproduces the
  real communication behavior the example is built to expose.

So CPX is a fine substitute for running the *scaling mechanics*, but tell
attendees that intra-chip CPX scaling understates RCCL cost — the interesting
communication behavior only appears once the rank count crosses physical APUs
(i.e. on the 24-GPU `PPAC_MI300A_CPX` node).

## Next steps

- **[`README_rccl_optimization.md`](README_rccl_optimization.md)** — hands-on
  exercises that optimize the RCCL all-reduce by editing `main.py` directly
  (bf16 gradient compression, `NCCL_ALGO`/`PROTO`/channels, DDP bucketing/overlap).
- **[`README_compute_optimization.md`](README_compute_optimization.md)** — hands-on
  exercises that optimize per-GPU compute by editing `main.py` directly (bf16
  autocast, `channels_last`, `cudnn.benchmark`, `torch.compile`, fused optimizer).
- **[`benchmarks/`](benchmarks/README_benchmark.md)** — the rigorous study:
  automated sweep drivers (`benchmarks/ddp_bench_sweep.sh`), optimization levers
  (`--channels-last`, `--amp`, `--compile`), the required MI300A/RCCL settings,
  measured results, batch jobs, the pure-RCCL bandwidth micro-benchmark, and how
  this compares to the other distributed examples.
- **[`profiling/`](profiling/PROFILING.md)** — splitting a step into compute vs.
  communication with torch.profiler, rocprofv3, and rocprof-sys.
- **Self-contained runs** — `run_imagenet_uv.sh` (and `submit_imagenet_uv.batch`)
  build a disposable **uv** venv, clone the upstream example, warm, sweep, and
  clean up automatically. See [`benchmarks/README_benchmark.md`](benchmarks/README_benchmark.md).
