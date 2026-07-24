# Compute optimization exercises (hands-on upstream edits)

A set of small, self-contained exercises for speeding up the **per-GPU compute**
of the upstream PyTorch minGPT-DDP example (the GPT block's attention, GEMMs, and
layernorms). As with the [RCCL exercises](README_rccl_optimization.md), you apply
every change **by hand** in the upstream code (`mingpt/trainer.py`,
`mingpt/model.py`) so you can see where each optimization lives and copy the
pattern into your own workload.

These target *compute*, not communication. Use a fixed **2-GPU** baseline to keep
the DDP all-reduce constant while you vary the compute — that isolates
kernel/precision/overhead effects from the gradient all-reduce. (Once you've
picked the fast compute settings, rerun the multi-GPU scaling sweep in
[`README.md`](README.md) to see the combined effect.)

> Keep it simple: do one edit, rerun the **same** baseline command, compare
> `step_s`, then move on. Undo an edit before the next unless a section says to
> stack them.

---

## Setup: make the trainer editable and measurable

Inside your allocation, with PyTorch loaded (see [`README.md`](README.md) §1):

```bash
git clone --depth=1 https://github.com/pytorch/examples.git ~/pytorch_examples
cd ~/pytorch_examples/distributed/minGPT-ddp
pip install -r requirements.txt
cp mingpt/trainer.py mingpt/trainer.orig.py    # so you can diff / reset
```

Apply the **measurement patch** from [`README.md`](README.md) §3a (the timed
`RESULT` line). You do **not** need `PROFILE=1` here — compute exercises are read
from `step_s` and `peak_mem_mb`.

### Baseline command (reuse this for every exercise)

```bash
torchrun --standalone --nproc_per_node=2 mingpt/main.py \
  trainer_config.max_epochs=1 data_config.truncate=0.2 \
  trainer_config.use_amp=False 2>&1 | tee run.log
grep RESULT run.log
```

Watch **`step_s`** in the `RESULT` line — average per-step time (lower is better).
Throughput is `tokens_per_s` on the same line; `peak_mem_mb` is the per-GPU peak
(bf16 also *reduces* memory, which lets you push a bigger `trainer_config.batch_size`).

---

## Section 1 — Lower-precision math

MI300A has fast bf16; using it for the matmul- and attention-heavy GPT block is
the single biggest compute win.

### 1a. bf16 autocast

The trainer already has an autocast path gated by `trainer_config.use_amp`, but it
uses **fp16** (with a `GradScaler`). bf16 has enough dynamic range that no scaler
is needed and it avoids fp16 overflow. Find, in `Trainer._run_batch`
(`mingpt/trainer.py`):

```python
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=self.device_type, dtype=torch.float16, enabled=(self.config.use_amp)):
```

Change the dtype to bf16:

```python
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=self.device_type, dtype=torch.bfloat16, enabled=(self.config.use_amp)):
```

Then run with `trainer_config.use_amp=True`:

```bash
torchrun --standalone --nproc_per_node=2 mingpt/main.py \
  trainer_config.max_epochs=1 data_config.truncate=0.2 trainer_config.use_amp=True
```

**Expect:** `step_s` drops substantially and `peak_mem_mb` drops too. (The
`GradScaler` still runs under `use_amp` but is a no-op for bf16; you can leave it.)

### 1b. Reduced-precision float32 matmuls

For any ops left in fp32, allow the lower-precision (TF32-style) matmul path. Add
once in `ddp_setup()` in `mingpt/main.py`, right after `init_process_group`:

```python
    init_process_group(backend=backend)
    torch.set_float32_matmul_precision("high")
```

**Expect:** a smaller gain than 1a on its own; mainly helps the fp32 baseline.
Harmless to leave on with 1a.

---

## Section 2 — Fuse kernels & cut launch overhead

### 2a. `torch.compile`

Captures the GPT block into a graph and fuses kernels (GEMM + bias + GELU,
attention epilogues), cutting Python/launch overhead. Find the DDP wrap in
`Trainer.__init__` (`mingpt/trainer.py`):

```python
        self.model = DDP(self.model, device_ids=[self.local_rank])
```

Compile the model right after it (same indentation):

```python
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.model = torch.compile(self.model)
```

**Expect:** the **first** step is much slower (one-time compile — it lands in the
warm-up window, so it does not pollute `step_s`), then per-step `step_s` improves.
Try `torch.compile(self.model, mode="max-autotune")` for more aggressive tuning at
a longer compile cost. (This is the same lever as the `COMPILE=1` gate in
[`README.md`](README.md) §3b and `ddp_gpt_bench.py --compile`.)

### 2b. Fused optimizer

A fused optimizer does the whole parameter update in one kernel instead of one per
tensor, cutting launch overhead. Find, in `create_optimizer` (`mingpt/model.py`):

```python
    optimizer = torch.optim.AdamW(optim_groups, lr=opt_config.learning_rate, betas=(0.9, 0.95))
```

Add `fused=True`:

```python
    optimizer = torch.optim.AdamW(optim_groups, lr=opt_config.learning_rate, betas=(0.9, 0.95), fused=True)
```

(The trainer already clears gradients with `zero_grad(set_to_none=True)`, so no
change is needed there.)

**Expect:** a small `step_s` improvement, biggest when steps are short (after the
other optimizations shrink compute).

---

## Section 3 — Attention fast path (advanced)

The GPT block uses `torch.nn.MultiheadAttention` (`mingpt/model.py`,
`MultiheadAttentionLayer`). PyTorch routes it to a fused scaled-dot-product-
attention (flash / memory-efficient) kernel **only when conditions are met** — in
particular contiguous bf16/fp16 inputs and a compatible mask. Two things unlock it:

- **Run bf16 (Section 1a).** fp32 attention usually falls back to the math kernel;
  bf16 lets the fused kernel engage.
- **The additive float `attn_mask`** passed here
  (`self.mask[0, 0, :seq_size, :seq_size]`) can push MHA onto the slower path. For
  a causal LM you can instead pass `is_causal=True` and drop the explicit mask.
  Find, in `MultiheadAttentionLayer.forward`:

```python
        y = self.attn(x, x, x, attn_mask=self.mask[0, 0, :seq_size, :seq_size])[0]
```

and try the causal fast path:

```python
        y = self.attn(x, x, x, is_causal=True, need_weights=False)[0]
```

**Expect:** on builds where AOTriton flash attention is available and engaged, a
`step_s` improvement (and lower attention memory); otherwise the number is
unchanged. Verify which kernel ran with `rocprofv3 --kernel-trace` (look for a
flash/efficient-attention kernel vs. a plain `bmm`/softmax sequence). This is
model-correctness-sensitive — keep the original mask if outputs matter.

---

## Recommended stack

For this GPT block on MI300A, the high-value combination is **bf16 autocast (1a)
+ torch.compile (2a)**. Apply both, then rerun the multi-GPU sweep from
[`README.md`](README.md): compute gets faster, which also makes the fixed gradient
all-reduce a larger share of each (now shorter) step — the reason the
[RCCL exercises](README_rccl_optimization.md) matter more after you optimize compute.

## What moves what

| Exercise | Edit location | Watch |
|---|---|---|
| 1a bf16 autocast | `dtype=` in `trainer._run_batch` | `step_s` ↓↓, `peak_mem_mb` ↓ |
| 1b matmul precision | after `init_process_group` in `main.py` | `step_s` (fp32 path) |
| 2a `torch.compile` | after `DDP(...)` in `trainer.py` | `step_s` ↓ (slow 1st step) |
| 2b fused optimizer | `AdamW(...)` in `model.py` | `step_s` (small) |
| 3a attention fast path | `MultiheadAttentionLayer.forward` in `model.py` | `step_s` (build-dependent) |

**Reset between exercises:** undo your edit, or restore the original files (this
also removes the Setup instrumentation, so re-apply the README §3a patch afterward).

**Apply to your own workload:** the portable patterns are the bf16 autocast
context around your forward/loss (1a), wrapping your model in `torch.compile`
(2a), and a fused optimizer (2b). Always re-measure — gains depend on model shape
and how compute- vs. memory-bound your kernels are.
