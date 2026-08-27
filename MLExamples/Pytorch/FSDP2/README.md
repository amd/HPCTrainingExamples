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
saves).

This README is a **hands-on walk-through**: you clone the upstream
`pytorch/examples/distributed/FSDP2` example and apply a handful of small,
visible patches that take it from the stock training loop to an **instrumented**
one (throughput + peak memory), then add **profiling** and **optimizations** on
top. The patches are deliberately tiny so the progression is clear in a session.

> For the polished, all-in-one benchmark (`fsdp2_bench.py`, which bakes in every
> patch below plus a sweep driver and batch jobs), see
> **[`benchmarks/README_benchmark.md`](benchmarks/README_benchmark.md)**. For the
> full profiling guide, see **[`profiling/PROFILING.md`](profiling/PROFILING.md)**.

> On ROCm, PyTorch's `nccl` backend is **librccl**; all `NCCL_*` variables apply.
> FSDP2 (`fully_shard`) requires **PyTorch >= 2.5** (ideally a recent release).

## Repository layout

| Path | Purpose |
|------|---------|
| `README.md` (this file) | Hands-on: clone the upstream example, patch in instrumentation + optimizations |
| `fsdp2_speedup.sh` | Summarizes the manual sweep (`run_*.log` -> throughput / peak-mem / speedup table) |
| [`README_rccl_optimization.md`](README_rccl_optimization.md) | Hands-on exercises: optimize the all-gather / reduce-scatter (bf16, `NCCL_*`, prefetch, reshard) |
| [`README_compute_optimization.md`](README_compute_optimization.md) | Hands-on exercises: optimize per-GPU compute (bf16, `torch.compile`, fused optimizer, SDPA) |
| [`benchmarks/`](benchmarks/README_benchmark.md) | Polished `fsdp2_bench.py`, automated sweep, measured results, batch jobs |
| [`profiling/`](profiling/PROFILING.md) | Compute-vs-communication attribution (torch.profiler, rocprofv3, rocprof-sys, ...) |

## 1. Get an allocation and load PyTorch

FSDP2 needs at least 2 GPUs; grab a few:

```bash
salloc --gpus=8 --ntasks=1 --time=01:00:00
module load rocm openmpi pytorch      # or set up a venv/container as in ../mnist
```

Verify FSDP2 is importable and the GPUs are visible:

```bash
python3 -c 'from torch.distributed.fsdp import fully_shard; print("FSDP2 OK")'
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

> **MI300A node requirement.** The node must be booted with `iommu=pt`
> (`grep -o 'iommu=pt' /proc/cmdline`) so RCCL uses direct xGMI P2P; otherwise the
> FSDP2 collectives hang. The upstream `example.py` already passes `device_id=` to
> `init_process_group` (also required). Details in
> [`benchmarks/README_benchmark.md`](benchmarks/README_benchmark.md).

## 2. Get the upstream example

```bash
git clone --depth=1 https://github.com/pytorch/examples.git ~/pytorch_examples
cd ~/pytorch_examples/distributed/FSDP2
```

Run it once, unmodified, to see the stock 10-step training loop shard the toy
transformer (it prints the model and writes a `checkpoints/` folder):

```bash
torchrun --standalone --nproc_per_node=2 example.py
```

The stock `example.py` only trains and checkpoints — it reports **no timing,
throughput, or memory**. The patches below add exactly that.

## 3. Patch the example: add measurement, then optimizations

Each edit below is a single `sed` you can read before running. Apply them in
order to `example.py` (keep a copy of the original if you want to diff:
`cp example.py example.orig.py`).

### 3a. Enlarge the toy model so sharding/scaling is measurable

The upstream toy model (`dim=16`, seq 64) is too small to show a memory or
communication trend. Bump it to a realistic transformer (16 layers, dim 1024,
16 heads, seq 512, per-GPU batch 8):

```bash
sed -i 's/    vocab_size = 1024/    vocab_size = 32000/' example.py
sed -i 's/    batch_size = 32/    batch_size = 8/'       example.py
sed -i 's/    seq_len = 64/    seq_len = 512/'           example.py
sed -i 's/        n_layers=10,/        n_layers=16,/'    example.py
sed -i 's/        n_heads=4,/        n_heads=16,\n        dim=1024,/' example.py
```

### 3b. Time a fixed window and report throughput + peak memory

Warm up a few steps, then time the next `_iters` steps and print a `RESULT` line
(step time, global tokens/s, per-GPU peak memory). This is the core measurement.

```bash
# turn the 10-step loop into warmup + timed iters (and pre-declare the profiler)
sed -i 's/^    for _ in range(10):/    _warmup, _iters = 5, 50\n    _prof = None\n    for _i in range(_warmup + _iters):/' example.py

# start the CUDA-event timer (and reset peak memory) at the first timed step
sed -i '/^    for _i in range(_warmup + _iters):/a\
        if _i == _warmup:\
            torch.cuda.synchronize(); torch.distributed.barrier()\
            torch.cuda.reset_peak_memory_stats(device)\
            _t0 = torch.cuda.Event(enable_timing=True); _t1 = torch.cuda.Event(enable_timing=True)\
            _t0.record()\
            if os.environ.get("PROFILE") == "1":\
                import torch.profiler as _tp\
                _prof = _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA])\
                _prof.start()' example.py

# stop the timer after the loop, print RESULT (and RCCL time if PROFILE=1)
sed -i '/^    checkpointer.save(model, optim)/i\
    _t1.record(); torch.cuda.synchronize()\
    _step_s = _t0.elapsed_time(_t1) / _iters / 1e3\
    _tokens = batch_size * seq_len\
    _ws = torch.distributed.get_world_size()\
    _peak_mb = torch.cuda.max_memory_allocated(device) / 1e6\
    if torch.distributed.get_rank() == 0:\
        _mp = "bf16" if args.mixed_precision else "fp32"\
        print(f"RESULT world_size={_ws} step_s={_step_s:.4f} "\
              f"tokens_per_s={_tokens * _ws / _step_s:.0f} "\
              f"peak_mem_mb={_peak_mb:.0f} precision={_mp}")\
    if _prof is not None:\
        _prof.stop()\
        _rccl_ms = sum(e.self_device_time_total for e in _prof.key_averages()\
                       if "nccl" in e.key.lower()) / 1e3\
        if torch.distributed.get_rank() == 0:\
            print(f"RCCL_TOTAL_MS {_rccl_ms:.3f} world_size={_ws}")' example.py
```

The `PROFILE=1`-gated block starts a `torch.profiler` over the timed window and
sums the on-GPU time of the `nccl*` collective kernels (the FSDP2 all-gather +
reduce-scatter). It is off by default so the plain timing runs are not perturbed.

### 3c. Add a `torch.compile` optimization (env-gated)

Wrap the sharded model in `torch.compile` (graph capture + kernel fusion) when
`COMPILE=1`, so you can compare against the eager baseline without a second copy
of the script:

```bash
sed -i '/^    optim = torch.optim.Adam/i\
    if os.environ.get("COMPILE") == "1":\
        model = torch.compile(model)' example.py
```

> Mixed precision and explicit prefetching are **already built into** the
> upstream example as `--mixed-precision` and `--explicit-prefetching`, so those
> optimizations need no patch — you toggle them on the command line in step 6.

Sanity-check the patched file, then run one instrumented pass:

```bash
python3 -c 'import ast; ast.parse(open("example.py").read()); print("syntax OK")'
torchrun --standalone --nproc_per_node=2 example.py
# -> RESULT world_size=2 step_s=... tokens_per_s=... peak_mem_mb=... precision=fp32
```

## 4. Confirm the RCCL topology

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
  torchrun --standalone --nproc_per_node=2 example.py 2>&1 \
  | grep -E 'NCCL|Ring|Channel|Tree' | head -40
```

You want `via P2P`/`[xGMI]` (on-fabric), not `via SHM`/`via PCI` (host-staged).

## 5. Scaling sweep by hand (the RCCL + memory study)

Run the instrumented example once per GPU count, teeing each to `run_<N>.log`.
Remove `checkpoints/` first so a run never reloads a checkpoint from a different
world size:

```bash
rm -rf checkpoints
for n in 2 4 8; do
  rm -rf checkpoints
  torchrun --standalone --nproc_per_node=$n example.py 2>&1 | tee run_$n.log
done
```

Summarize with the helper shipped in this directory:

```bash
/path/to/HPCTrainingExamples/MLExamples/Pytorch/FSDP2/fsdp2_speedup.sh
```

Illustrative output:

```
GPUs   step_s       tok_per_s      peak_mem_MB    speedup    eff
2      0.1641       49931          7086           1.00       100%
4      0.1566       104615         6285           2.10       105%
8      0.1400       228000         3200           4.57       114%
```

Read two things together:

- **peak_mem_MB drops** as GPUs increase — the sharding win (what lets FSDP2
  train models too large to fit with DDP).
- **throughput scales** but the all-gather/reduce-scatter overhead means the
  per-step time does not shrink perfectly linearly.

## 6. Optimizations — compare against the baseline

Re-run the sweep (or a single GPU count) with each optimization and diff the
`RESULT` lines:

```bash
# bf16 params / fp32 reduce -- halves the all-gather byte volume (built-in flag)
for n in 2 4 8; do rm -rf checkpoints; \
  torchrun --standalone --nproc_per_node=$n example.py --mixed-precision 2>&1 | tee run_$n.log; done
./fsdp2_speedup.sh

# explicit prefetching of the next layers' all-gather (built-in flag)
torchrun --standalone --nproc_per_node=4 example.py --explicit-prefetching

# torch.compile the sharded model (the patch from step 3c)
COMPILE=1 torchrun --standalone --nproc_per_node=4 example.py --mixed-precision
```

> **ROCm 7.2.x bf16 gotcha.** The bf16 path can stall for minutes in hipBLASLt on
> ROCm 7.2.x. Run on ROCm 6.4.3, or `export TORCH_BLAS_PREFER_HIPBLASLT=0`. See
> [`../common/hipblaslt-notes.md`](../common/hipblaslt-notes.md).

## 7. Profiling — see the sharded collectives

Turn on the profiler patch (step 3b) to get the total RCCL kernel time; it grows
with GPU count because there is more all-gather/reduce-scatter to do:

```bash
PROFILE=1 torchrun --standalone --nproc_per_node=4 example.py 2>&1 | tee prof_4.log
grep RCCL_TOTAL_MS prof_4.log
```

For a framework-independent kernel trace (no code change), run the whole thing
under `rocprofv3` and look for the FSDP2 collectives:

```bash
rocprofv3 --kernel-trace --stats --truncate-kernels -- \
  torchrun --standalone --nproc_per_node=4 example.py 2>&1 | grep -iE 'AllGather|ReduceScatter'
```

`ncclDevKernel_AllGather_*` and `ncclDevKernel_ReduceScatter_*` are the FSDP2
collectives, distinct from DDP's single `AllReduce`. For the full guide
(torch.profiler tables, TensorBoard, rocprof-compute roofline, rocprof-sys
timeline), see [`profiling/PROFILING.md`](profiling/PROFILING.md).

## 8. Cleanup

```bash
rm -rf checkpoints run_*.log prof_*.log
deactivate 2>/dev/null || true        # if you used a venv
```

## Next steps

- **[`README_rccl_optimization.md`](README_rccl_optimization.md)** — hands-on
  exercises that optimize the FSDP2 all-gather / reduce-scatter by editing
  `example.py` directly (bf16 params/grads, `NCCL_ALGO`/`PROTO`/channels,
  `reshard_after_forward`, explicit prefetching).
- **[`README_compute_optimization.md`](README_compute_optimization.md)** —
  hands-on exercises that optimize per-GPU compute by editing `example.py`
  directly (bf16 params, `torch.compile`, fused optimizer, SDPA backend).
- **[`benchmarks/README_benchmark.md`](benchmarks/README_benchmark.md)** — the
  polished `fsdp2_bench.py` bakes in every patch above (timing, `--rccl-time`,
  `--profile`, `--flops`, `--mixed-precision`, `--compile`, zero-copy input
  staging) and adds an automated sweep driver, larger-model sweeps, the required
  MI300A/RCCL settings, measured results, and batch jobs.
- **[`profiling/PROFILING.md`](profiling/PROFILING.md)** — splitting a step into
  compute vs. communication with torch.profiler, the DeepSpeed FlopsProfiler,
  TensorBoard, rocprofv3, rocprof-compute (roofline), and rocprof-sys (timeline).
