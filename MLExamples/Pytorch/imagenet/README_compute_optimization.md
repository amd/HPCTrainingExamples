# Compute optimization exercises (hands-on `main.py` edits)

A set of small, self-contained exercises for speeding up the **per-GPU compute**
of the upstream PyTorch imagenet example (ResNet50 forward/backward). As with the
[RCCL exercises](README_rccl_optimization.md), you apply every change **by hand in
`main.py`** so you can see where each optimization lives and copy the pattern into
your own workload.

These target *compute*, not communication, so the exercises use a **single-GPU**
baseline — that isolates kernel/precision/overhead effects from the RCCL
all-reduce. (Once you've picked the fast compute settings, rerun the multi-GPU
scaling sweep in [`README.md`](README.md) to see the combined effect.)

> Keep it simple: do one edit, rerun the **same** baseline command, compare the
> per-step time, then move on. Undo an edit before the next unless a section says
> to stack them.

---

## Setup: make `main.py` editable and measurable

Inside your allocation, with the PyTorch module loaded and the MIOpen cache warmed
(see [`README.md`](README.md) §2-3):

```bash
git clone --depth=1 https://github.com/pytorch/examples.git
cd examples/imagenet
```

Keep each run short — find `data_time.update(time.time() - end)` in the training
loop of `train()` and add a break right after it:

```python
        # measure data loading time
        data_time.update(time.time() - end)
        if i >= 100: break
```

### Baseline command (reuse this for every exercise)

```bash
HIP_VISIBLE_DEVICES=0 python main.py -a resnet50 --dummy \
  --dist-url 'tcp://127.0.0.1:23456' --dist-backend nccl \
  --multiprocessing-distributed --world-size 1 --rank 0 -b 128 -p 20 --epochs 1
```

Watch the **`Time`** value in the `Epoch:` lines — the average per-step time (lower
is better). Throughput is roughly **img/s = 128 / Time**. Optionally watch memory
in another shell with `rocm-smi` (bf16 + channels_last also *reduce* memory, which
lets you push a bigger batch later).

---

## Section 1 — Lower-precision math

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

**Expect:** per-step `Time` drops substantially; peak memory drops too.

### 1b. Reduced-precision float32 matmuls

For any ops left in fp32, allow the lower-precision (bf16-accumulate) matmul path.
Add once near the top of `main_worker()`, just after `args.gpu = gpu`:

```python
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    torch.set_float32_matmul_precision("high")
```

**Expect:** a smaller gain than 1a on its own; mainly helps runs that stay in
fp32. Harmless to leave on with 1a.

---

## Section 2 — Memory layout & kernel selection

### 2a. `channels_last` (NHWC) memory format

CDNA convolutions prefer NHWC. Converting the model and the input batch lets MIOpen
pick faster kernels. This is **two** edits.

Edit 1 — right after the model is created in `main_worker()`:

```python
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        model = model.to(memory_format=torch.channels_last)
```

Edit 2 — in the `train()` loop, convert the input batch. Find:

```python
        images = images.to(device, non_blocking=True)
```

and change it to:

```python
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
```

**Expect:** per-step `Time` drops, and it stacks well with bf16 (1a) — that
combination is the recommended default.

### 2b. Autotune convolution kernels — `cudnn.benchmark`

Lets the backend (MIOpen on ROCm) search for the fastest conv algorithm for the
fixed input shape and cache it. Add near the top of `main_worker()`:

```python
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    cudnn.benchmark = True
```

**Expect:** a one-time search cost on the first steps, then a faster steady-state
`Time`. (`MIOPEN_FIND_MODE=FAST` and the warmed cache from `README.md` §3 keep the
search cheap.) Note this is incompatible with the deterministic `--seed` path.

---

## Section 3 — Fuse kernels & cut launch overhead

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

**Expect:** the **first** step is much slower (one-time compile), then per-step
`Time` improves. Try `torch.compile(model, mode="max-autotune")` for more
aggressive tuning at a longer compile cost.

### 3b. Fused optimizer + cheaper `zero_grad`

A fused optimizer does the whole parameter update in one kernel instead of one per
tensor, cutting launch overhead.

Edit 1 — the optimizer in `main_worker()`. Find:

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

Edit 2 — in the `train()` loop, make the gradient clear cheaper. Find
`optimizer.zero_grad()` and change it to:

```python
        optimizer.zero_grad(set_to_none=True)
```

**Expect:** a small `Time` improvement, biggest when steps are short (small model
or after the other optimizations shrink compute).

---

## Recommended stack

For ResNet50 on MI300A, the high-value combination is **bf16 autocast (1a) +
channels_last (2a) + torch.compile (3a)**. Apply all three, then rerun the
multi-GPU sweep from [`README.md`](README.md): compute gets faster, which also
makes the RCCL all-reduce a larger share of each (now shorter) step — the reason
the [RCCL exercises](README_rccl_optimization.md) matter more after you optimize
compute.

## What moves what

| Exercise | Edit location in `main.py` | Watch |
|---|---|---|
| 1a bf16 autocast | forward/loss in `train()` loop | `Time` ↓↓, memory ↓ |
| 1b matmul precision | top of `main_worker()` | `Time` (fp32 ops) |
| 2a channels_last | after model create + input in loop | `Time` ↓ (with 1a) |
| 2b `cudnn.benchmark` | top of `main_worker()` | `Time` ↓ after warm-up |
| 3a `torch.compile` | after `DistributedDataParallel(...)` | `Time` ↓ (slow 1st step) |
| 3b fused optimizer | SGD args + `zero_grad` in loop | `Time` (small) |

**Reset between exercises:** undo your edit, or `git checkout -- main.py` (this
also removes the Setup break, so re-add it afterward).

**Apply to your own workload:** the portable patterns are the autocast context
around your forward/loss (1a), the `channels_last` conversion of model + inputs
(2a), and wrapping your model in `torch.compile` (3a). Always re-measure — gains
depend on model shape and how compute- vs. memory-bound your kernels are.
