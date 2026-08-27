# Compute optimization exercises (hands-on `main.py` edits)

Here we present a set of small, self-contained tips for speeding up the per-GPU
compute of the upstream PyTorch imagenet example (ResNet50 forward/backward). We apply
every change by hand in `main.py`, so we can see exactly where each optimization lives
and reuse the same pattern in our own workloads.

These build on the scaling study in [`README.md`](README.md) but target *compute*, not
communication, so the exercises use a single-GPU baseline: that isolates
kernel/precision/overhead effects from the RCCL all-reduce. Once we have picked the fast
compute settings, we rerun the multi-GPU scaling sweep in [`README.md`](README.md) to
see the combined effect.

---

## Setup

These exercises assume §1-8 of [`README.md`](README.md) have already been completed. If
not, work through them first.

### Baseline command (reuse this to assess the impact of each tip)

```bash
HIP_VISIBLE_DEVICES=0 python main.py -a resnet50 --dummy \
  --dist-url 'tcp://127.0.0.1:23456' --dist-backend nccl \
  --multiprocessing-distributed --world-size 1 --rank 0 -b 128 -p 20 --epochs 1
```

We keep it simple: do one edit, rerun the same baseline command, compare the per-step
time, then move to the next. Undo an edit before the next one unless a section says to
stack them, so each contribution stays isolated.

We watch the `Time` value in the `Epoch:` lines: the average per-step time (lower is
better). Throughput is roughly `img/s = 128 / Time`. We can also watch memory in another
shell with `rocm-smi` (bf16 and channels_last also reduce memory, which leaves room for
a bigger batch later).

---

## Section 1: Lower-precision math

MI300A has fast bf16; using it for the matmul/conv-heavy work is the single biggest
compute win.

### 1a. bf16 autocast (AMP)

Casts the forward pass and loss to bf16. bf16 has enough dynamic range that no
`GradScaler` is needed.

Find, in the training loop of `train()`:

```python
        # compute output
        output = model(images)
        loss = criterion(output, target)
```

Wrap it in an autocast context:

```python
        # compute output
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(images)
            loss = criterion(output, target)
```

Expect: per-step `Time` drops substantially, and peak memory drops too.

### 1b. Reduced-precision float32 matmuls

For any ops left in fp32, allow the lower-precision (bf16-accumulate) matmul path.
Add once near the top of `main_worker()`, just after `args.gpu = gpu`:

```python
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    torch.set_float32_matmul_precision("high")
```

Expect: a smaller gain than 1a on its own, mainly for runs that stay in fp32. Harmless
to leave on together with 1a.

---

## Section 2: Memory layout and kernel selection

### 2a. `channels_last` (NHWC) memory format

CDNA convolutions prefer NHWC. Converting the model and the input batch lets MIOpen
pick faster kernels. This is two edits.

Edit 1: right after the model is created in `main_worker()`:

```python
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        model = model.to(memory_format=torch.channels_last)
```

Edit 2: in the `train()` loop, convert the input batch. Find:

```python
        images = images.to(device, non_blocking=True)
```

and change it to:

```python
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
```

Expect: per-step `Time` drops, and it stacks well with bf16 (1a): that combination is
the recommended default.

### 2b. Autotune convolution kernels: `cudnn.benchmark`

Lets the backend (MIOpen on ROCm) search for the fastest conv algorithm for the
fixed input shape and cache it. Add near the top of `main_worker()`:

```python
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    cudnn.benchmark = True
```

Expect: a one-time search cost on the first steps, then a faster steady-state `Time`.
(`MIOPEN_FIND_MODE=FAST` and the warmed cache from `README.md` §3 keep the search cheap.)
Note this is incompatible with the deterministic `--seed` path.

---

## Section 3: Fuse kernels and cut launch overhead

### 3a. `torch.compile`

Captures the model into a graph and fuses kernels, cutting Python/launch overhead.

Find the DDP line in `main_worker()`:

```python
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
```

Compile the model right after it (same indentation):

```python
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                model = torch.compile(model)
```

Expect: the first step is much slower (one-time compile), then per-step `Time` improves.
Try `torch.compile(model, mode="max-autotune")` for more aggressive tuning at a longer
compile cost.

### 3b. Fused optimizer + cheaper `zero_grad`

A fused optimizer does the whole parameter update in one kernel instead of one per
tensor, cutting launch overhead.

Edit 1: the optimizer in `main_worker()`. Find:

```python
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
```

Add `fused=True`:

```python
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, fused=True)
```

Edit 2: in the `train()` loop, make the gradient clear cheaper. Find
`optimizer.zero_grad()` and change it to:

```python
        optimizer.zero_grad(set_to_none=True)
```

Expect: a small `Time` improvement, biggest when steps are short (small model or after
the other optimizations shrink compute).

---

## Recommended stack

For ResNet50 on MI300A, the high-value combination is bf16 autocast (1a), channels_last
(2a), and torch.compile (3a). We apply all three, then rerun the multi-GPU sweep from
[`README.md`](README.md): compute gets faster, which also makes the RCCL all-reduce a
larger share of each (now shorter) step. That is why the
[RCCL exercises](README_rccl_optimization.md) matter more once compute is optimized.

## What moves what

| Exercise | Edit location in `main.py` | Watch |
|---|---|---|
| 1a bf16 autocast | forward/loss in `train()` loop | `Time` much lower, memory lower |
| 1b matmul precision | top of `main_worker()` | `Time` (fp32 ops) |
| 2a channels_last | after model create + input in loop | `Time` lower (with 1a) |
| 2b `cudnn.benchmark` | top of `main_worker()` | `Time` lower after warm-up |
| 3a `torch.compile` | after `DistributedDataParallel(...)` | `Time` lower (slow 1st step) |
| 3b fused optimizer | SGD args + `zero_grad` in loop | `Time` (small) |

Apply to your own workload: the portable patterns are the autocast context around the
forward/loss (1a), the `channels_last` conversion of model and inputs (2a), and wrapping
the model in `torch.compile` (3a). Always re-measure: gains depend on model shape and how
compute- or memory-bound the kernels are.

---

## Measured impact: batch study (`torch.compile`, 1 GPU)

The edits above are packaged as a self-contained SLURM script,
`run_imagenet_uv_compute_opt.sbatch`, so the throughput impact is reproducible with a
single submission. The script builds a disposable `uv` venv, clones the upstream
example, applies the source-code instrumentation, warms the MIOpen cache, then runs the
model twice on one SPX GPU at batch 512: once eager and once through `torch.compile`.

```bash
# eager vs torch.compile (1 GPU, SPX, b=512)
sbatch run_imagenet_uv_compute_opt.sbatch

# read the summary the job appends to its .out file
sed -n '/=== compute optimization/,$p' run_imagenet_uv_spx-<jobid>.out
```

On `PPAC_MI300A_SPX` the job takes about 14 minutes end to end (`--time=02:00:00` is
generous headroom); the wall time is dominated by the venv build, the upstream clone,
MIOpen warmup, and the one-time `torch.compile` graph build rather than the ~100 timed
steps.

We report throughput as a steady-state median, not a running average. The script's
`summarize()` collects the instantaneous per-step `Time`, drops the first (warmup /
one-time graph-compile) step, and takes the median; `img/s` is the global batch divided
by that median. Averaging `Time` over the full 100-iteration cap instead would fold the
tens-of-seconds first compile step into the mean and make `torch.compile` look like a
regression.

The script prints (steady-state median over its 100-iteration capped run, ~9 samples
after the warmup step is dropped):

| Metric | eager | `torch.compile` | Impact |
|---|---|---|---|
| Steady-state median step | 0.416 s | 0.384 s | -7.7 % (faster) |
| Steady-state img/s | 1232 | 1335 | +8.3 % |
| Peak memory (`PEAK_MEM_MB`) | 44874 MB | 42329 MB | -5.7 % |

A 100-iteration window samples only the fast early steps, so it reads slightly
optimistic. Re-measuring the same median over a full epoch (2503 steps, ~116 samples per
phase) gives the sustained figures eager 0.445 s / 1151 img/s and compile 0.421 s /
1216 img/s, a steadier +5.4 %. Either way the ranking is the same.

Takeaway: once the one-time compile is amortized, `torch.compile` gives a solid
throughput gain (~5-8 %) and lower peak memory for ResNet-50 on a single MI300A. Try
`torch.compile(model, mode="max-autotune")` for a larger gain at a longer compile cost.

When resources allow, prefer longer runs. The study is capped at 100 iterations so it
finishes quickly and stays a polite neighbor on a shared system. The steady-state median
(first step dropped) already keeps the one-time graph build out of the headline number,
but 100 iterations still read slightly optimistic versus the sustained rate. With more
compute and fewer users, raise the iteration cap (`if i >= 100: break` in `main.py`) or
run full epochs so the compile cost amortizes and the median approaches the true
sustained throughput, which is the number to quote for real training.
