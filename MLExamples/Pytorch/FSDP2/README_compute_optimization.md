# Compute optimization exercises (hands-on `example.py` edits)

A set of small, self-contained exercises for speeding up the **per-GPU compute**
of the upstream PyTorch FSDP2 example (the transformer's GEMMs, attention, and
layernorms). As with the [RCCL exercises](README_rccl_optimization.md), you apply
every change **by hand in `example.py`** so you can see where each optimization
lives and copy the pattern into your own workload.

These target *compute*, not communication. FSDP2 needs ≥2 GPUs, so use a fixed
**2-GPU** baseline to keep the sharding communication constant while you vary the
compute — that isolates kernel/precision/overhead effects from the all-gather /
reduce-scatter. (Once you've picked the fast compute settings, rerun the
multi-GPU scaling sweep in [`README.md`](README.md) to see the combined effect.)

> Keep it simple: do one edit, rerun the **same** baseline command, compare
> `step_s`, then move on. Undo an edit before the next unless a section says to
> stack them.

---

## Setup: make `example.py` editable and measurable

Inside your allocation, with PyTorch loaded (see [`README.md`](README.md) §1):

```bash
git clone --depth=1 https://github.com/pytorch/examples.git ~/pytorch_examples
cd ~/pytorch_examples/distributed/FSDP2
cp example.py example.orig.py            # so you can diff / reset
```

Apply the **measurement patches** from [`README.md`](README.md) §3a–3b (enlarge
the model + print the timed `RESULT` line). You do **not** need `PROFILE=1` here —
compute exercises are read from `step_s` and `peak_mem_mb`.

### Baseline command (reuse this for every exercise)

```bash
rm -rf checkpoints
torchrun --standalone --nproc_per_node=2 example.py 2>&1 | tee run.log
grep RESULT run.log
```

Watch **`step_s`** in the `RESULT` line — average per-step time (lower is better).
Throughput is `tokens_per_s` on the same line; `peak_mem_mb` is the per-GPU peak
(bf16 also *reduces* memory, which lets you push a bigger `batch_size` later).

---

## Section 1 — Lower-precision math

MI300A has fast bf16; using it for the matmul- and attention-heavy transformer is
the single biggest compute win.

### 1a. bf16 parameters (`MixedPrecisionPolicy`)

Running the sharded parameters in bf16 makes every GEMM and the attention run in
bf16. Find the `fsdp_kwargs` block in `main()`:

```python
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
```

Enable it unconditionally (keep the gradient reduce in fp32 for stability):

```python
    fsdp_kwargs = {}
    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
```

**Expect:** `step_s` drops substantially and `peak_mem_mb` drops too. (This is the
same knob as [RCCL 1a](README_rccl_optimization.md#1a-bf16-parameters-and-gradients-mixedprecisionpolicy),
viewed from the compute side: bf16 params also halve the all-gather bytes.)

### 1b. Reduced-precision float32 matmuls

For any ops left in fp32, allow the lower-precision (TF32-style) matmul path. Add
once in `main()`, right after the seed:

```python
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
```

**Expect:** a smaller gain than 1a on its own; mainly helps the fp32 baseline.
Harmless to leave on with 1a.

---

## Section 2 — Fuse kernels & cut launch overhead

### 2a. `torch.compile`

Captures the sharded model into a graph and fuses kernels (GEMM + bias + GELU,
attention epilogues), cutting Python/launch overhead. Find where the optimizer is
created in `main()`:

```python
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
```

Compile the model on the line **above** it:

```python
    model = torch.compile(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
```

**Expect:** the **first** step is much slower (one-time compile — it lands in the
warm-up window, so it does not pollute `step_s`), then per-step `step_s` improves.
Try `torch.compile(model, mode="max-autotune")` for more aggressive tuning at a
longer compile cost. (This is the same lever as the `COMPILE=1` gate in
[`README.md`](README.md) §3c and `fsdp2_bench.py --compile`.)

### 2b. Fused optimizer + cheaper `zero_grad`

A fused optimizer does the whole parameter update in one kernel instead of one
per tensor, cutting launch overhead. Find:

```python
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
```

Add `fused=True`:

```python
    optim = torch.optim.Adam(model.parameters(), lr=1e-2, fused=True)
```

Then make the gradient clear cheaper. Find `optim.zero_grad()` in the training
loop and change it to:

```python
        optim.zero_grad(set_to_none=True)
```

**Expect:** a small `step_s` improvement, biggest when steps are short (after the
other optimizations shrink compute).

---

## Section 3 — Attention kernel selection (advanced)

The model's `Attention` uses `F.scaled_dot_product_attention` (SDPA), which
dispatches to a math, memory-efficient, or flash kernel. You can pin the backend
to compare. Add the import at the top of `example.py`:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
```

Then wrap the forward in the training loop. Find:

```python
        loss = model(x).sum()
```

and pin the fused kernel:

```python
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            loss = model(x).sum()
```

**Expect:** on builds where AOTriton flash attention is available, a `step_s`
improvement (and lower attention memory); otherwise SDPA falls back to
`EFFICIENT_ATTENTION`/`MATH` and the number is unchanged. Try
`SDPBackend.EFFICIENT_ATTENTION` as the portable fallback to compare.

---

## Recommended stack

For this transformer on MI300A, the high-value combination is **bf16 params (1a)
+ torch.compile (2a)**. Apply both, then rerun the multi-GPU sweep from
[`README.md`](README.md): compute gets faster, which also makes the FSDP2
all-gather/reduce-scatter a larger share of each (now shorter) step — the reason
the [RCCL exercises](README_rccl_optimization.md) matter more after you optimize
compute.

## What moves what

| Exercise | Edit location in `example.py` | Watch |
|---|---|---|
| 1a bf16 params | `fsdp_kwargs` `MixedPrecisionPolicy` | `step_s` ↓↓, `peak_mem_mb` ↓ |
| 1b matmul precision | after `torch.manual_seed(0)` | `step_s` (fp32 path) |
| 2a `torch.compile` | before `optim =` | `step_s` ↓ (slow 1st step) |
| 2b fused optimizer | `Adam(...)` args + `zero_grad` in loop | `step_s` (small) |
| 3a SDPA backend | import + `with sdpa_kernel(...)` around forward | `step_s` (build-dependent) |

**Reset between exercises:** undo your edit, or `cp example.orig.py example.py`
(this also removes the Setup instrumentation, so re-apply the README §3 patches
afterward).

**Apply to your own workload:** the portable patterns are the bf16
`MixedPrecisionPolicy` on `fully_shard` (1a), wrapping the model in
`torch.compile` (2a), and a fused optimizer with `set_to_none=True` (2b). Always
re-measure — gains depend on model shape and how compute- vs. memory-bound your
kernels are.
