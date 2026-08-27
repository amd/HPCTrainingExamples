# minGPT-DDP: LLM-shaped RCCL all-reduce scaling

README.md from `HPCTrainingExamples/MLExamples/Pytorch/minGPT-ddp` in the Training Examples repository

This example studies **RCCL** communication for a **transformer (GPT)** trained
with `DistributedDataParallel` (DDP). It complements the
[imagenet](../imagenet) example: same DDP all-reduce collective and same
measurement methodology, but the gradient tensors are **transformer-shaped and
much larger**, so the all-reduce is a bigger fraction of each step — closer to
what you see training real language models.

This README is a **hands-on walk-through**: you clone the upstream
`pytorch/examples/distributed/minGPT-ddp` example and apply a handful of small,
visible patches that take its **real training loop** (`mingpt/trainer.py`, a
character-level GPT on tinyshakespeare) from the stock version to an
**instrumented** one (throughput + peak memory), then add **profiling** and
**optimizations** on top. The patches are deliberately tiny so the progression is
clear in a session.

> For the polished, synthetic-data benchmark (`ddp_gpt_bench.py`, which needs no
> dataset and adds the `no_sync()` direct comm-isolation `comm_pct`), plus the
> sweep driver, measured results, and batch jobs, see
> **[`benchmarks/README_benchmark.md`](benchmarks/README_benchmark.md)**. For the
> full profiling guide, see **[`profiling/PROFILING.md`](profiling/PROFILING.md)**.

> On ROCm, PyTorch's `nccl` backend is **librccl**; all `NCCL_*` variables apply.

## Repository layout

| Path | Purpose |
|------|---------|
| `README.md` (this file) | Hands-on: clone the upstream example, patch in instrumentation + optimizations |
| `mingpt_speedup.sh` | Summarizes the manual sweep (`run_*.log` -> throughput / peak-mem / speedup / RCCL table) |
| [`README_rccl_optimization.md`](README_rccl_optimization.md) | Hands-on exercises: optimize the gradient all-reduce (bf16 comm, `NCCL_*`, DDP bucketing/overlap) |
| [`README_compute_optimization.md`](README_compute_optimization.md) | Hands-on exercises: optimize per-GPU compute (bf16, `torch.compile`, fused optimizer, SDPA) |
| [`benchmarks/`](benchmarks/README_benchmark.md) | Polished `ddp_gpt_bench.py` (synthetic data, `no_sync()`), automated sweep, measured results, batch jobs |
| [`profiling/`](profiling/PROFILING.md) | Compute-vs-communication attribution (torch.profiler, rocprofv3, rocprof-sys, ...) |

## 1. Get an allocation, load PyTorch, and clone the example

DDP here needs at least 2 GPUs; grab a few:

```bash
salloc --gpus=8 --ntasks=1 --time=01:00:00
module load rocm openmpi pytorch      # or set up a venv/container as in ../mnist

git clone --depth=1 https://github.com/pytorch/examples.git ~/pytorch_examples
cd ~/pytorch_examples/distributed/minGPT-ddp
pip install -r requirements.txt       # hydra, fsspec, boto3 for the real trainer
```

Confirm the GPUs are visible:

```bash
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

> **MI300A node requirement.** The node must be booted with `iommu=pt`
> (`grep -o 'iommu=pt' /proc/cmdline`) so RCCL uses direct xGMI P2P; otherwise the
> gradient all-reduce hangs. Details in
> [`benchmarks/README_benchmark.md`](benchmarks/README_benchmark.md).

## 2. Run the stock trainer once

The upstream training job trains a character-level GPT on tinyshakespeare (hydra
config in `mingpt/gpt2_train_cfg.yaml`). Run one short epoch to see it work:

```bash
torchrun --standalone --nproc_per_node=2 mingpt/main.py \
  trainer_config.max_epochs=1 data_config.truncate=0.2
```

The stock trainer prints periodic `Loss` lines but **no timing, throughput, or
memory**. The patches below add exactly that.

## 3. Patch the trainer: add measurement, then an optimization

Each edit is a single `sed` you can read before running. Apply them to
`mingpt/trainer.py` (keep a copy to diff/reset:
`cp mingpt/trainer.py mingpt/trainer.orig.py`).

### 3a. Time a fixed window and report throughput + peak memory

Warm up a few steps, time the next `_iters` steps, print a `RESULT` line (step
time, global tokens/s, per-GPU peak memory), then stop early so a "run" is quick.
The `PROFILE=1`-gated block adds a `torch.profiler` that sums the `nccl*` kernel
time as `RCCL_TOTAL_MS`.

```bash
# init the timer/profiler state just before the training loop in _run_epoch()
sed -i '/^        for iter, (source, targets) in enumerate(dataloader):/i\
        _warmup, _iters = 3, 20\
        _prof = None; _t0 = _t1 = None' mingpt/trainer.py

# start/stop the timer inside the loop (train only), print RESULT, then break
sed -i '/^            step_type = "Train" if train else "Eval"/i\
            if train and iter == _warmup:\
                torch.cuda.synchronize(); torch.distributed.barrier()\
                torch.cuda.reset_peak_memory_stats(self.local_rank)\
                _t0 = torch.cuda.Event(enable_timing=True); _t1 = torch.cuda.Event(enable_timing=True); _t0.record()\
                if os.environ.get("PROFILE") == "1":\
                    import torch.profiler as _tp\
                    _prof = _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA]); _prof.start()\
            if train and _t0 is not None and iter == _warmup + _iters:\
                _t1.record(); torch.cuda.synchronize()\
                _step_s = _t0.elapsed_time(_t1) / _iters / 1e3\
                _ws = torch.distributed.get_world_size()\
                _tok = source.size(0) * source.size(1)\
                if self.global_rank == 0:\
                    print(f"RESULT world_size={_ws} step_s={_step_s:.4f} "\
                          f"tokens_per_s={_tok * _ws / _step_s:.0f} "\
                          f"peak_mem_mb={torch.cuda.max_memory_allocated(self.local_rank)/1e6:.0f}")\
                if _prof is not None:\
                    _prof.stop()\
                    _rccl_ms = sum(e.self_device_time_total for e in _prof.key_averages()\
                                   if "nccl" in e.key.lower()) / 1e3\
                    if self.global_rank == 0:\
                        print(f"RCCL_TOTAL_MS {_rccl_ms:.3f} world_size={_ws}")\
                break' mingpt/trainer.py
```

### 3b. Add a `torch.compile` optimization (env-gated)

Wrap the model in `torch.compile` when `COMPILE=1`, right after DDP wraps it, so
you can compare against eager without a second copy of the script:

```bash
sed -i '/^        self.model = DDP(self.model, device_ids=\[self.local_rank\])/a\
        if os.environ.get("COMPILE") == "1":\
            self.model = torch.compile(self.model)' mingpt/trainer.py
```

Sanity-check the patched file, then run one instrumented pass:

```bash
python3 -c 'import ast; ast.parse(open("mingpt/trainer.py").read()); print("syntax OK")'
torchrun --standalone --nproc_per_node=2 mingpt/main.py \
  trainer_config.max_epochs=1 data_config.truncate=0.2
# -> RESULT world_size=2 step_s=... tokens_per_s=... peak_mem_mb=...
```

## 4. Confirm the RCCL topology

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
  torchrun --standalone --nproc_per_node=2 mingpt/main.py \
  trainer_config.max_epochs=1 data_config.truncate=0.2 2>&1 \
  | grep -E 'NCCL|Ring|Channel|Tree' | head -40
```

Prefer `via ... [xGMI]` / `P2P` (on-fabric) over `via SHM`/`via PCI`.

## 5. Scaling sweep by hand (the RCCL study)

Run the instrumented trainer once per GPU count, teeing each to `run_<N>.log`.
`truncate=0.2` keeps enough batches per rank even at 8 GPUs; DDP here needs ≥2:

```bash
for n in 2 4 8; do
  torchrun --standalone --nproc_per_node=$n mingpt/main.py \
    trainer_config.max_epochs=1 data_config.truncate=0.2 2>&1 | tee run_$n.log
done
```

Summarize with the helper shipped in this directory:

```bash
/path/to/HPCTrainingExamples/MLExamples/Pytorch/minGPT-ddp/mingpt_speedup.sh
```

Illustrative output:

```
GPUs   step_s       tok_per_s      peak_mem_MB    speedup    eff      rccl_ms
2      0.1061       77196          5200           1.00       100%     -
4      0.1039       157722         5200           2.04       102%     -
8      0.1080       303000         5200           3.93       98%      -
```

As GPUs increase, the gap of `speedup`/`eff` from ideal is the RCCL gradient
all-reduce cost. **Amplify the signal** with a bigger model (bigger gradients):

```bash
for n in 2 4 8; do
  torchrun --standalone --nproc_per_node=$n mingpt/main.py \
    trainer_config.max_epochs=1 data_config.truncate=0.2 \
    gpt_config.n_layer=12 gpt_config.n_head=12 gpt_config.n_embd=768 2>&1 | tee run_$n.log
done
./mingpt_speedup.sh
```

## 6. Optimizations — compare against the baseline

The upstream trainer already has a built-in mixed-precision path
(`trainer_config.use_amp`); toggle it and the `torch.compile` patch from step 3b:

```bash
# fp32 baseline
torchrun --standalone --nproc_per_node=4 mingpt/main.py \
  trainer_config.max_epochs=1 data_config.truncate=0.2 trainer_config.use_amp=False

# mixed precision (built-in autocast)
torchrun --standalone --nproc_per_node=4 mingpt/main.py \
  trainer_config.max_epochs=1 data_config.truncate=0.2 trainer_config.use_amp=True

# torch.compile the model (the step 3b patch)
COMPILE=1 torchrun --standalone --nproc_per_node=4 mingpt/main.py \
  trainer_config.max_epochs=1 data_config.truncate=0.2 trainer_config.use_amp=True
```

> **ROCm 7.2.x bf16 gotcha.** The bf16 path can stall for minutes in hipBLASLt on
> ROCm 7.2.x. Run on ROCm 6.4.3, or `export TORCH_BLAS_PREFER_HIPBLASLT=0`. See
> [`../common/hipblaslt-notes.md`](../common/hipblaslt-notes.md).

See [`README_compute_optimization.md`](README_compute_optimization.md) and
[`README_rccl_optimization.md`](README_rccl_optimization.md) for the full set of
by-hand optimization exercises.

## 7. Profiling — measure the all-reduce

Turn on the profiler patch (step 3a) to get the total RCCL kernel time; it grows
with GPU count because there is more gradient all-reduce to do:

```bash
PROFILE=1 torchrun --standalone --nproc_per_node=4 mingpt/main.py \
  trainer_config.max_epochs=1 data_config.truncate=0.2 2>&1 | tee prof_4.log
grep RCCL_TOTAL_MS prof_4.log
```

For a framework-independent kernel trace (no code change), run under `rocprofv3`:

```bash
rocprofv3 --kernel-trace --stats --truncate-kernels -- \
  torchrun --standalone --nproc_per_node=4 mingpt/main.py \
  trainer_config.max_epochs=1 data_config.truncate=0.2 2>&1 | grep -i AllReduce
```

RCCL collectives appear as `ncclDevKernel_AllReduce_*`. For the full guide
(torch.profiler tables, TensorBoard, rocprof-compute roofline, rocprof-sys
timeline), see [`profiling/PROFILING.md`](profiling/PROFILING.md).

## 8. Cleanup

```bash
rm -f run_*.log prof_*.log *snapshot*.pt
```

## Next steps

- **[`README_rccl_optimization.md`](README_rccl_optimization.md)** — hands-on
  exercises that optimize the DDP gradient all-reduce by editing the upstream
  code directly (bf16 gradient-compression comm hook, `NCCL_ALGO`/`PROTO`/channels,
  DDP bucketing/overlap).
- **[`README_compute_optimization.md`](README_compute_optimization.md)** —
  hands-on exercises that optimize per-GPU compute by editing the code directly
  (bf16 autocast, `torch.compile`, fused optimizer, SDPA backend).
- **[`benchmarks/README_benchmark.md`](benchmarks/README_benchmark.md)** — the
  polished `ddp_gpt_bench.py` drives the same GPT on **synthetic tokens** (no
  dataset), adds the `no_sync()` direct `comm_pct` measurement, and provides an
  automated sweep driver, measured results, and batch jobs.
- **[`profiling/PROFILING.md`](profiling/PROFILING.md)** — splitting a step into
  compute vs. communication with torch.profiler, the DeepSpeed FlopsProfiler,
  TensorBoard, rocprofv3, rocprof-compute (roofline), and rocprof-sys (timeline).
