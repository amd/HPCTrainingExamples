# RCCL optimization exercises (hands-on `example.py` edits)

A set of small, self-contained exercises for optimizing the **FSDP2
communication** — the parameter **all-gather** (forward + backward) and the
gradient **reduce-scatter** (backward) — in the upstream PyTorch FSDP2 example.
You apply every change **by hand in `example.py`** — not by flipping shell flags
— so you can see exactly *where* each optimization lives and copy the same
pattern into your own sharded workload.

These build on the scaling study in [`README.md`](README.md). The workshop nodes
all have **`iommu=pt`** set, so RCCL already uses the direct xGMI / Infinity
Fabric path — you don't need to touch the transport for correctness, only to
*tune* it.

> Unlike DDP's single `all_reduce` (see [imagenet](../imagenet/README_rccl_optimization.md)),
> FSDP2's collectives are woven into forward/backward and cannot be toggled off
> with `no_sync()`. So we read the RCCL kernel time and the scaling behavior
> instead of subtracting a "no-comm" step.

> Keep it simple: do one edit, rerun the **same** baseline command, compare the
> numbers, then move to the next. Undo an edit before the next one unless a
> section says to stack them.

---

## Setup: make `example.py` editable and measurable

Inside your allocation, with PyTorch loaded (see [`README.md`](README.md) §1):

```bash
git clone --depth=1 https://github.com/pytorch/examples.git ~/pytorch_examples
cd ~/pytorch_examples/distributed/FSDP2
cp example.py example.orig.py            # so you can diff / reset
```

Apply the **measurement patches** from [`README.md`](README.md) §3a–3b: they
enlarge the toy model so the collectives are measurable and add the timed
`RESULT` line plus the `PROFILE=1`-gated `torch.profiler` block that prints
`RCCL_TOTAL_MS` (the summed on-GPU time of the `nccl*` collective kernels). The
`RCCL_TOTAL_MS` number is the one these exercises move.

### Baseline command (reuse this for every exercise)

FSDP2 shards across ranks, so use **≥2 GPUs**. 4 GPUs gives a clearer RCCL signal:

```bash
rm -rf checkpoints
PROFILE=1 torchrun --standalone --nproc_per_node=4 example.py 2>&1 | tee run.log
grep -E 'RESULT|RCCL_TOTAL_MS' run.log
```

Record two numbers each time:

- **`RCCL_TOTAL_MS`** — total all-gather + reduce-scatter time (lower is better).
- **`step_s`** in the `RESULT` line — average per-step time (lower is better; this
  is what improves when communication is *hidden* behind compute).

> The RCCL signal is small when all ranks share one MI300A (on-package fabric is
> nearly free). For a stronger signal, rerun with more GPUs — especially the
> `PPAC_MI300A_CPX` 12- and 24-GPU cases, where the collectives cross physical APUs.

---

## Section 1 — Reduce the bytes on the wire

The fewer bytes RCCL moves, the cheaper the all-gather and reduce-scatter. This
is usually the biggest single win.

### 1a. bf16 parameters and gradients (`MixedPrecisionPolicy`)

The all-gather moves parameters in their storage dtype and the reduce-scatter
moves gradients in `reduce_dtype`. Casting both to bf16 **halves** the bytes on
the wire. Find the `fsdp_kwargs` block in `main()`:

```python
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
```

Set the policy unconditionally so you can A/B `reduce_dtype`:

```python
    fsdp_kwargs = {}
    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,   # try float32 (safe) vs bfloat16 (half the reduce bytes)
    )
```

**Expect:** `RCCL_TOTAL_MS` drops toward half with `param_dtype=bfloat16` (smaller
all-gather). Switching `reduce_dtype` to `bfloat16` cuts the reduce-scatter bytes
too, at a small numerical-accuracy cost — keep `float32` if convergence matters.

---

## Section 2 — Tune the RCCL transport / algorithm

RCCL reads its `NCCL_*` settings **when the communicator is built** — i.e. at
`init_process_group(...)`. So these must be set in `example.py` *before* that
call. That ordering is the whole point of doing it here rather than exporting a
shell variable after the fact.

Find this block near the top of `main()`:

```python
    backend = torch.distributed.get_default_backend_for_device(device)
    torch.distributed.init_process_group(backend=backend, device_id=device)
```

Add your setting(s) on the line **above** `init_process_group` (`os` is already
imported), e.g.:

```python
    backend = torch.distributed.get_default_backend_for_device(device)
    os.environ["NCCL_ALGO"] = "Tree"
    torch.distributed.init_process_group(backend=backend, device_id=device)
```

### 2a. Collective algorithm — `NCCL_ALGO`

```python
    os.environ["NCCL_ALGO"] = "Tree"   # try "Ring" (default) vs "Tree"
```

`Ring` maximizes bandwidth for the large all-gather buffers; `Tree` cuts latency
and often wins once a collective crosses APUs (12/24-GPU CPX).

### 2b. Wire protocol — `NCCL_PROTO`

```python
    os.environ["NCCL_PROTO"] = "LL128"   # try "Simple", "LL", "LL128"
```

`LL128` is usually the sweet spot for medium messages on high-bandwidth coherent
links; `Simple` favors the largest messages.

### 2c. Channel count — `NCCL_MIN_NCHANNELS`

```python
    os.environ["NCCL_MIN_NCHANNELS"] = "8"   # more channels = more parallelism
```

More channels drive the copy with more compute units, raising effective
bandwidth on big collectives until it saturates. (`NCCL_MAX_NCHANNELS` caps it.)

**Expect (2a–2c):** `RCCL_TOTAL_MS` shifts up or down; the best choice depends on
message size and whether ranks span APUs. Change **one** variable at a time.

---

## Section 3 — Overlap and reshape the collectives (FSDP2 knobs)

These don't change the bytes; they change how well the all-gather/reduce-scatter
is *hidden* behind compute, and how often parameters are re-gathered.

### 3a. Keep parameters gathered after forward — `reshard_after_forward`

By default FSDP2 frees (reshards) each layer's parameters after the forward pass
and **all-gathers them again** in backward. Setting `reshard_after_forward=False`
keeps them resident, removing the backward re-gather — less communication, more
memory. Find the sharding loop in `main()`:

```python
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
```

and add the argument:

```python
    for layer in model.layers:
        fully_shard(layer, reshard_after_forward=False, **fsdp_kwargs)
    fully_shard(model, reshard_after_forward=False, **fsdp_kwargs)
```

**Expect:** `RCCL_TOTAL_MS` drops (no backward all-gather) while `peak_mem_mb`
rises — the classic FSDP2 memory-vs-communication trade. This is the opposite
direction from sharding harder; use it only when you have spare memory.

### 3b. Prefetch the next layers' all-gather — explicit prefetching

Overlapping a layer's all-gather with the previous layer's compute *hides* the
communication. The example already wires this up behind `--explicit-prefetching`
(`set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch`). Turn it
on, and tune the depth. Find:

```python
    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)
```

Run with the flag (and sweep the depth 1 → 2 → 3):

```bash
PROFILE=1 torchrun --standalone --nproc_per_node=4 example.py --explicit-prefetching
```

**Expect:** per-step `step_s` drops as the all-gather overlaps compute, even
though `RCCL_TOTAL_MS` (the bytes) is unchanged. Prefetching too many layers
raises peak memory (more parameters gathered at once) for diminishing overlap.

---

## What moves what

| Exercise | Edit location in `example.py` | Watch |
|---|---|---|
| 1a bf16 params/grads | `fsdp_kwargs` `MixedPrecisionPolicy` | `RCCL_TOTAL_MS` ↓ (~half) |
| 2a `NCCL_ALGO` | before `init_process_group` | `RCCL_TOTAL_MS` |
| 2b `NCCL_PROTO` | before `init_process_group` | `RCCL_TOTAL_MS` |
| 2c `NCCL_MIN_NCHANNELS` | before `init_process_group` | `RCCL_TOTAL_MS` |
| 3a `reshard_after_forward=False` | `fully_shard(...)` args | `RCCL_TOTAL_MS` ↓, `peak_mem_mb` ↑ |
| 3b explicit prefetching | `--explicit-prefetching` + depth | step `step_s` |

**Reset between exercises:** undo your edit, or `cp example.orig.py example.py`
(this also removes the Setup instrumentation, so re-apply the README §3 patches
afterward).

**Apply to your own workload:** the portable levers are the `MixedPrecisionPolicy`
on `fully_shard` (Section 1), the `reshard_after_forward` memory/comm trade and
explicit prefetching (Section 3), and the `NCCL_*` transport settings set before
`init_process_group` (Section 2, workload- and topology-dependent — always
re-measure). See [`README_compute_optimization.md`](README_compute_optimization.md)
for the per-GPU compute side.
