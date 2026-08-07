# RCCL optimization exercises (hands-on `main.py` edits)

Here we present a set of small, self-contained tips for optimizing the **RCCL gradient
all-reduce** in the upstream PyTorch imagenet example. You will apply every change **by
hand in `main.py`** so you can see exactly where
each optimization lives and copy the same pattern into your own workload as needed.

These build on the scaling study in [`README.md`](README.md). We assume that the nodes
you are running on all have **`iommu=pt`** set (it is the case for the nodes on AAC6), 
so RCCL already uses the direct xGMI / Infinity Fabric path — you don't need to touch
 the transport for correctness, only to tune it.

---

## Setup

These exercises assume you have already worked through **§1-8 of
[`README.md`](README.md)**. If not, please go through those steps and come back.

### Baseline command (reuse this to assess the impact of each tip)

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 python main.py -a resnet50 --dummy \
  --dist-url 'tcp://127.0.0.1:23456' --dist-backend nccl \
  --multiprocessing-distributed --world-size 1 --rank 0 -b 512 -p 20 --epochs 1
```

We suggest you keep it simple: do one edit, rerun the **same** baseline command, compare the
numbers, then move to the next. Undo an edit before the next one unless a
section says to stack them, so you can isolate each contribution.

Record two numbers each time:

- **`RCCL_TOTAL_MS`** — total RCCL communication time (lower is better).
- the **`Time`** value in the `Epoch:` lines — the average per-step time (lower is
  better; this is what improves when communication is *hidden* behind compute).

> The RCCL signal is small on a single MI300A (on-package fabric is nearly free).
> For a stronger signal, rerun with more GPUs — especially the `PPAC_MI300A_CPX`
> 12- and 24-GPU cases, where the all-reduce crosses physical APUs.

---

## Section 1 — Reduce the bytes on the wire

The fewer bytes RCCL moves, the cheaper the all-reduce. This is usually the
biggest single win.

### 1a. bf16 gradient compression

`--amp` casts compute to bf16, but DDP still all-reduces **fp32** gradients. A
communication hook halves the bytes by compressing gradients to bf16 for the
all-reduce only.

Find the DDP line in `main_worker()`:

```python
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
```

Add the hook right below it (same indentation):

```python
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_hooks
                model.register_comm_hook(None, ddp_hooks.bf16_compress_hook)
```

**Expect:** `RCCL_TOTAL_MS` drops toward half; per-step `Time` improves when the
run is communication-bound (more GPUs / across APUs). Try `fp16_compress_hook` for
comparison.

---

## Section 2 — Tune the RCCL transport / algorithm

RCCL reads its `NCCL_*` settings **when the communicator is built** — i.e. at
`dist.init_process_group(...)`. So these must be set in `main.py` before that
call. That ordering is the whole point of doing it here rather than exporting a
shell variable after the fact.

Find this block in `main_worker()`:

```python
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
```

Add your setting(s) on the line **above** it (8-space indent), e.g.:

```python
        os.environ["NCCL_ALGO"] = "Tree"
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
```

### 2a. Collective algorithm — `NCCL_ALGO`

```python
        os.environ["NCCL_ALGO"] = "Tree"   # try "Ring" (default) vs "Tree"
```

`Ring` maximizes bandwidth for ResNet50's large gradient buckets; `Tree` cuts
latency and often wins once the all-reduce crosses APUs (12/24-GPU CPX).

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

**Expect (2a-2c):** `RCCL_TOTAL_MS` shifts up or down; the best choice depends on
message size and whether ranks span APUs. Change **one** variable at a time.

---

## Section 3 — Overlap and shrink the collectives (DDP knobs)

These don't change the bytes; they change how well the all-reduce is hidden
behind the backward pass and how many separate collectives are launched. Watch the
per-step **`Time`** here more than `RCCL_TOTAL_MS`.

All three edit the same DDP line in `main_worker()`:

```python
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
```

### 3a. Avoid a gradient copy — `gradient_as_bucket_view=True`

```python
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], gradient_as_bucket_view=True)
```

Lets DDP reduce gradients in place instead of copying them into buckets.

### 3b. Bucket size — `bucket_cap_mb`

```python
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], bucket_cap_mb=100)
```

Default is 25 MB. On fast fabric, larger buckets mean fewer, larger, more
bandwidth-efficient all-reduces (trade-off: slightly less compute/comm overlap).
Sweep 25 → 50 → 100.

### 3c. Static graph — `static_graph=True`

```python
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], static_graph=True)
```

Tells DDP the graph is fixed each step, cutting per-iteration bookkeeping (and
enabling better overlap). Do **not** set `find_unused_parameters=True` — it adds a
synchronization.

You can stack all three:

```python
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu],
                    gradient_as_bucket_view=True, bucket_cap_mb=100, static_graph=True)
```

**Expect:** per-step `Time` drops as the same all-reduce overlaps better with
compute; `RCCL_TOTAL_MS` (the bytes) stays about the same.

---

## What moves what

| Exercise | Edit location in `main.py` | Watch |
|---|---|---|
| 1a bf16 hook | after `DistributedDataParallel(...)` | `RCCL_TOTAL_MS` ↓ (~half) |
| 2a `NCCL_ALGO` | before `init_process_group` | `RCCL_TOTAL_MS` |
| 2b `NCCL_PROTO` | before `init_process_group` | `RCCL_TOTAL_MS` |
| 2c `NCCL_MIN_NCHANNELS` | before `init_process_group` | `RCCL_TOTAL_MS` |
| 3a `gradient_as_bucket_view` | `DistributedDataParallel(...)` args | step `Time` |
| 3b `bucket_cap_mb` | `DistributedDataParallel(...)` args | step `Time` |
| 3c `static_graph` | `DistributedDataParallel(...)` args | step `Time` |


**Apply to your own workload:** the two lines that matter everywhere are the
communication hook (Section 1, right after you wrap your model in DDP) and the DDP
constructor arguments (Section 3). The `NCCL_*` settings (Section 2) are workload-
and topology-dependent — always re-measure to be safe.
