# RCCL optimization exercises (hands-on `main.py` edits)

Here we present a set of small, self-contained tips for optimizing the RCCL gradient
all-reduce in the upstream PyTorch imagenet example. We apply every change by hand in
`main.py`, so we can see exactly where each optimization lives and reuse the same pattern
in our own workloads.

These build on the scaling study in [`README.md`](README.md). We assume the nodes all
have `iommu=pt` set (as they do on AAC6), so RCCL already uses the direct xGMI / Infinity
Fabric path: we do not need to touch the transport for correctness, only to tune it.

---

## Setup

These exercises assume §1-8 of [`README.md`](README.md) have already been completed. If
not, work through them first.

### Baseline command (reuse this to assess the impact of each tip)

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 python main.py -a resnet50 --dummy \
  --dist-url 'tcp://127.0.0.1:23456' --dist-backend nccl \
  --multiprocessing-distributed --world-size 1 --rank 0 -b 512 -p 20 --epochs 1
```

We keep it simple: do one edit, rerun the same baseline command, compare the numbers,
then move to the next. Undo an edit before the next one unless a section says to stack
them, so each contribution stays isolated.

We record two numbers each time:

- `RCCL_TOTAL_MS`: total RCCL communication time (lower is better).
- the `Time` value in the `Epoch:` lines: the average per-step time (lower is better;
  this is what improves when communication is hidden behind compute).

> The RCCL signal is small within one SPX node: the all-reduce stays on the fast on-node
> Infinity Fabric, and a single APU is one SPX device, so on its own it has no all-reduce at
> all. For a stronger signal, scale up: the 12-/24-GPU `PPAC_MI300A_CPX` cases, or multiple
> nodes where the all-reduce crosses NICs over RDMA.

---

## Section 1: Reduce the bytes on the wire

The fewer bytes RCCL moves, the cheaper the all-reduce. This is usually the
biggest single win.

### 1a. bf16 gradient compression

`--amp` casts compute to bf16, but DDP still all-reduces fp32 gradients. A communication
hook halves the bytes by compressing gradients to bf16 for the all-reduce only.

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

Expect: `RCCL_TOTAL_MS` drops toward half, and per-step `Time` improves when the run is
communication-bound (more GPUs, or across APUs). Try `fp16_compress_hook` for comparison.

---

## Section 2: Tune the RCCL transport and algorithm

RCCL reads its `NCCL_*` settings when the communicator is built, at
`dist.init_process_group(...)`. So these must be set in `main.py` before that call. That
ordering is the whole point of doing it here rather than exporting a shell variable after
the fact.

Find this block in `main_worker()`:

```python
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
```

Add the setting(s) on the line above it (8-space indent), for example:

```python
        os.environ["NCCL_ALGO"] = "Tree"
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
```

### 2a. Collective algorithm: `NCCL_ALGO`

```python
        os.environ["NCCL_ALGO"] = "Tree"   # try "Ring" (default) vs "Tree"
```

`Ring` maximizes bandwidth for ResNet50's large gradient buckets; `Tree` cuts
latency and often wins once the all-reduce crosses APUs (12/24-GPU CPX).

### 2b. Wire protocol: `NCCL_PROTO`

```python
        os.environ["NCCL_PROTO"] = "LL128"   # try "Simple", "LL", "LL128"
```

`LL128` is usually the sweet spot for medium messages on high-bandwidth coherent
links; `Simple` favors the largest messages.

### 2c. Channel count: `NCCL_MIN_NCHANNELS`

```python
        os.environ["NCCL_MIN_NCHANNELS"] = "8"   # more channels = more parallelism
```

More channels drive the copy with more compute units, raising effective bandwidth
on big all-reduces until it saturates. (`NCCL_MAX_NCHANNELS` caps it.)

Expect (2a-2c): `RCCL_TOTAL_MS` shifts up or down; the best choice depends on message
size and whether ranks span APUs. Change one variable at a time.

---

## Section 3: Overlap and shrink the collectives (DDP knobs)

These don't change the bytes; they change how well the all-reduce is hidden behind the
backward pass and how many separate collectives are launched. Watch the per-step `Time`
here more than `RCCL_TOTAL_MS`.

All three edit the same DDP line in `main_worker()`:

```python
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
```

### 3a. Avoid a gradient copy: `gradient_as_bucket_view=True`

```python
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], gradient_as_bucket_view=True)
```

Lets DDP reduce gradients in place instead of copying them into buckets.

### 3b. Bucket size: `bucket_cap_mb`

```python
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], bucket_cap_mb=100)
```

Default is 25 MB. On fast fabric, larger buckets mean fewer, larger, more
bandwidth-efficient all-reduces (trade-off: slightly less compute/comm overlap).
Sweep 25, then 50, then 100.

### 3c. Static graph: `static_graph=True`

```python
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], static_graph=True)
```

Tells DDP the graph is fixed each step, cutting per-iteration bookkeeping (and enabling
better overlap). Do not set `find_unused_parameters=True`: it adds a synchronization.

You can stack all three:

```python
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu],
                    gradient_as_bucket_view=True, bucket_cap_mb=100, static_graph=True)
```

Expect: per-step `Time` drops as the same all-reduce overlaps better with compute;
`RCCL_TOTAL_MS` (the bytes) stays about the same.

---

## What moves what

| Exercise | Edit location in `main.py` | Watch |
|---|---|---|
| 1a bf16 hook | after `DistributedDataParallel(...)` | `RCCL_TOTAL_MS` lower (~half) |
| 2a `NCCL_ALGO` | before `init_process_group` | `RCCL_TOTAL_MS` |
| 2b `NCCL_PROTO` | before `init_process_group` | `RCCL_TOTAL_MS` |
| 2c `NCCL_MIN_NCHANNELS` | before `init_process_group` | `RCCL_TOTAL_MS` |
| 3a `gradient_as_bucket_view` | `DistributedDataParallel(...)` args | step `Time` |
| 3b `bucket_cap_mb` | `DistributedDataParallel(...)` args | step `Time` |
| 3c `static_graph` | `DistributedDataParallel(...)` args | step `Time` |


Apply to your own workload: the two lines that matter everywhere are the communication
hook (Section 1, right after we wrap the model in DDP) and the DDP constructor arguments
(Section 3). The `NCCL_*` settings (Section 2) are workload- and topology-dependent, so
always re-measure to be safe.

---

## Measured impact: batch study (NCCL algorithm / protocol / channels, 4 GPUs)

The `NCCL_*` levers above are packaged as a self-contained SLURM sweep,
`run_imagenet_uv_rccl_opt_sweep.sbatch`. It builds a disposable `uv` venv, clones the
upstream example, applies the source-code instrumentation, warms the MIOpen cache, then
runs one phase per swept value on 4 SPX GPUs at batch 512, holding the other two knobs at
their NCCL defaults.

```bash
# NCCL algorithm / protocol / channel-count sweep (4 GPUs, SPX, b=512)
sbatch run_imagenet_uv_rccl_opt_sweep.sbatch

# read the summary the job appends to its .out file
sed -n '/=== RCCL total time by NCCL algorithm/,$p' run_imagenet_uv_spx-<jobid>.out
```

On `PPAC_MI300A_SPX` the 11-phase sweep (2 algorithm + 3 protocol + 6 channel) takes
about 44 minutes end to end (`--time=08:00:00` is generous headroom); each phase re-pays
MIOpen/allocator warmup.

Each table reports `RCCL_TOTAL_MS` (defined above; lower is better) with only the swept
knob set:

Algorithm (protocol + channels = default):

| `NCCL_ALGO` | RCCL_TOTAL_MS |
|---|---|
| Tree | 42630 |
| Ring | 45848 |

Protocol (algorithm + channels = default):

| `NCCL_PROTO` | RCCL_TOTAL_MS |
|---|---|
| LL | 4111 |
| Simple | 10819 |
| LL128 | 37944 |

Channel count (algorithm + protocol = default):

| `NCCL_MIN/MAX_NCHANNELS` | RCCL_TOTAL_MS |
|---|---|
| 1 | 30821 |
| 8 | 37725 |
| 16 | 40134 |
| 2 | 41681 |
| 32 | 44286 |
| 4 | 45877 |

Takeaway: `NCCL_PROTO=LL` is the dominant lever here, ~9-10x less collective time than
the LL128/auto baseline for this ~102 MB gradient all-reduce; `Tree` edges out `Ring`
(~7 %); and on this on-node all-reduce fewer channels win (1 is best, extra channels add
CUDA-block overhead without a bandwidth payoff).

> On-node caveat. In SPX mode each APU is exposed as a single device, so a single APU has
> no all-reduce to perform. This sweep runs across the four APUs of one SPX node, where the
> collective stays on the node's Infinity Fabric and is small and fast, so these are relative
> signals on an on-node collective. The ranking (especially the protocol effect) grows with
> the number of ranks and, above all, once the collective leaves the node: rerun on the
> 12-/24-GPU `PPAC_MI300A_CPX` node (more ranks; see the CPX section of
> [`README.md`](README.md#sec-cpx)), and across multiple nodes where the all-reduce crosses
> NICs over RDMA, to see that behavior.

### NCCL/RCCL variable defaults

On ROCm, `librccl` honors the `NCCL_*` names. Unless they are set, RCCL auto-selects; the
sweep's "default" columns above reflect those auto choices:

| Variable | Default | Behavior when unset |
|---|---|---|
| `NCCL_ALGO` | auto | Internal performance model picks per collective/size & topology (typically Ring or Tree). |
| `NCCL_PROTO` | auto | Chooses per message size from the allowed set: `LL,LL128,Simple` on supported platforms (`LL,Simple` otherwise). LL for small, LL128 for medium, Simple for large messages. |
| `NCCL_MIN_NCHANNELS` | platform-dependent | Auto-tuned lower bound on channels (CUDA blocks) from topology. |
| `NCCL_MAX_NCHANNELS` | platform-dependent (max capped at 32 in recent NCCL/RCCL) | Auto-tuned upper bound; RCCL picks the actual count within `[min,max]`. |

> Upstream guidance is that these are best left unset: manual values only win for a
> specific message size / topology, which is exactly what this sweep demonstrates
> (`NCCL_PROTO=LL` beats the auto choice for this particular all-reduce size).
