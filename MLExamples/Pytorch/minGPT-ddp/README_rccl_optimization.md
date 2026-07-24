# RCCL optimization exercises (hands-on upstream edits)

A set of small, self-contained exercises for optimizing the **DDP gradient
all-reduce** in the upstream PyTorch minGPT-DDP example. You apply every change
**by hand** in the upstream code (`mingpt/trainer.py`, `mingpt/main.py`) — not by
flipping shell flags — so you can see exactly *where* each optimization lives and
copy the same pattern into your own workload.

These build on the scaling study in [`README.md`](README.md). The workshop nodes
all have **`iommu=pt`** set, so RCCL already uses the direct xGMI / Infinity
Fabric path — you don't need to touch the transport for correctness, only to
*tune* it.

> The transformer's gradients are large (≈498 MB/step for a 124M-param GPT), so
> the all-reduce is a **bigger share of each step** than for the ResNet in the
> [imagenet](../imagenet/README_rccl_optimization.md) example — the same DDP
> collective, but a more LLM-representative RCCL workload.

> Keep it simple: do one edit, rerun the **same** baseline command, compare the
> numbers, then move to the next. Undo an edit before the next one unless a
> section says to stack them.

---

## Setup: make the trainer editable and measurable

Inside your allocation, with PyTorch loaded (see [`README.md`](README.md) §1):

```bash
git clone --depth=1 https://github.com/pytorch/examples.git ~/pytorch_examples
cd ~/pytorch_examples/distributed/minGPT-ddp
pip install -r requirements.txt
cp mingpt/trainer.py mingpt/trainer.orig.py    # so you can diff / reset
```

Apply the **measurement patches** from [`README.md`](README.md) §3a: they add the
timed `RESULT` line plus the `PROFILE=1`-gated `torch.profiler` block that prints
`RCCL_TOTAL_MS` (the summed on-GPU time of the `nccl*` collective kernels). That
number is the one these exercises move.

### Baseline command (reuse this for every exercise)

DDP here needs ≥2 GPUs; 4 gives a clearer RCCL signal:

```bash
PROFILE=1 torchrun --standalone --nproc_per_node=4 mingpt/main.py \
  trainer_config.max_epochs=1 data_config.truncate=0.2 2>&1 | tee run.log
grep -E 'RESULT|RCCL_TOTAL_MS' run.log
```

Record two numbers each time:

- **`RCCL_TOTAL_MS`** — total gradient all-reduce time (lower is better).
- **`step_s`** in the `RESULT` line — average per-step time (lower is better; this
  is what improves when communication is *hidden* behind compute).

> The RCCL signal is small on a single MI300A (on-package fabric is nearly free).
> For a stronger signal, rerun with more GPUs — especially the `PPAC_MI300A_CPX`
> 12- and 24-GPU cases, where the all-reduce crosses physical APUs, or amplify it
> with a bigger model (`gpt_config.n_layer=12 gpt_config.n_embd=768`).

---

## Section 1 — Reduce the bytes on the wire

The fewer bytes RCCL moves, the cheaper the all-reduce. This is usually the
biggest single win.

### 1a. bf16 gradient compression

Autocast casts *compute* to lower precision, but DDP still all-reduces **fp32**
gradients. A communication hook halves the bytes by compressing gradients to
bf16 for the all-reduce only. Find the DDP wrap in `Trainer.__init__`
(`mingpt/trainer.py`):

```python
        # wrap with DDP. this step will synch model across all the processes.
        self.model = DDP(self.model, device_ids=[self.local_rank])
```

Register the hook right below it (same indentation):

```python
        # wrap with DDP. this step will synch model across all the processes.
        self.model = DDP(self.model, device_ids=[self.local_rank])
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_hooks
        self.model.register_comm_hook(None, ddp_hooks.bf16_compress_hook)
```

**Expect:** `RCCL_TOTAL_MS` drops toward half; per-step `step_s` improves when the
run is communication-bound (more GPUs / across APUs). Try `fp16_compress_hook`
for comparison.

---

## Section 2 — Tune the RCCL transport / algorithm

RCCL reads its `NCCL_*` settings **when the communicator is built** — i.e. at
`init_process_group(...)`. So these must be set *before* that call. In minGPT that
call lives in `ddp_setup()` in `mingpt/main.py`:

```python
    backend = torch.distributed.get_default_backend_for_device(device)
    init_process_group(backend=backend)
```

Add your setting(s) on the line **above** `init_process_group` (`os` is already
imported in `main.py`), e.g.:

```python
    backend = torch.distributed.get_default_backend_for_device(device)
    os.environ["NCCL_ALGO"] = "Tree"
    init_process_group(backend=backend)
```

### 2a. Collective algorithm — `NCCL_ALGO`

```python
    os.environ["NCCL_ALGO"] = "Tree"   # try "Ring" (default) vs "Tree"
```

`Ring` maximizes bandwidth for the large gradient buckets; `Tree` cuts latency
and often wins once the all-reduce crosses APUs (12/24-GPU CPX).

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

More channels drive the copy with more compute units, raising effective bandwidth
on big all-reduces until it saturates. (`NCCL_MAX_NCHANNELS` caps it.)

**Expect (2a–2c):** `RCCL_TOTAL_MS` shifts up or down; the best choice depends on
message size and whether ranks span APUs. Change **one** variable at a time.

---

## Section 3 — Overlap and shrink the collectives (DDP knobs)

These don't change the bytes; they change how well the all-reduce is *hidden*
behind the backward pass and how many separate collectives are launched. Watch the
per-step **`step_s`** here more than `RCCL_TOTAL_MS`.

All three edit the same DDP line in `Trainer.__init__` (`mingpt/trainer.py`):

```python
        self.model = DDP(self.model, device_ids=[self.local_rank])
```

### 3a. Avoid a gradient copy — `gradient_as_bucket_view=True`

```python
        self.model = DDP(self.model, device_ids=[self.local_rank],
                         gradient_as_bucket_view=True)
```

Lets DDP reduce gradients in place instead of copying them into buckets.

### 3b. Bucket size — `bucket_cap_mb`

```python
        self.model = DDP(self.model, device_ids=[self.local_rank],
                         bucket_cap_mb=100)
```

Default is 25 MB. On fast fabric, larger buckets mean fewer, larger, more
bandwidth-efficient all-reduces (trade-off: slightly less compute/comm overlap).
Sweep 25 → 50 → 100.

### 3c. Static graph — `static_graph=True`

```python
        self.model = DDP(self.model, device_ids=[self.local_rank],
                         static_graph=True)
```

Tells DDP the graph is fixed each step, cutting per-iteration bookkeeping (and
enabling better overlap). Do **not** set `find_unused_parameters=True` — it adds a
synchronization.

You can stack all three:

```python
        self.model = DDP(self.model, device_ids=[self.local_rank],
                         gradient_as_bucket_view=True, bucket_cap_mb=100,
                         static_graph=True)
```

**Expect:** per-step `step_s` drops as the same all-reduce overlaps better with
compute; `RCCL_TOTAL_MS` (the bytes) stays about the same.

---

## What moves what

| Exercise | Edit location | Watch |
|---|---|---|
| 1a bf16 hook | after `DDP(...)` in `trainer.py` | `RCCL_TOTAL_MS` ↓ (~half) |
| 2a `NCCL_ALGO` | before `init_process_group` in `main.py` | `RCCL_TOTAL_MS` |
| 2b `NCCL_PROTO` | before `init_process_group` in `main.py` | `RCCL_TOTAL_MS` |
| 2c `NCCL_MIN_NCHANNELS` | before `init_process_group` in `main.py` | `RCCL_TOTAL_MS` |
| 3a `gradient_as_bucket_view` | `DDP(...)` args in `trainer.py` | step `step_s` |
| 3b `bucket_cap_mb` | `DDP(...)` args in `trainer.py` | step `step_s` |
| 3c `static_graph` | `DDP(...)` args in `trainer.py` | step `step_s` |

**Reset between exercises:** undo your edit, or `cp mingpt/trainer.orig.py
mingpt/trainer.py` (this also removes the Setup instrumentation, so re-apply the
README §3a patch afterward).

**Apply to your own workload:** the two lines that matter everywhere are the
communication hook (Section 1, right after you wrap your model in DDP) and the DDP
constructor arguments (Section 3). The `NCCL_*` settings (Section 2) are workload-
and topology-dependent — always re-measure. See
[`README_compute_optimization.md`](README_compute_optimization.md) for the per-GPU
compute side.
