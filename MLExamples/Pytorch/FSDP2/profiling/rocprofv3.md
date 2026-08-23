# rocprofv3 — FSDP2 (RCCL device kernels + timeline)

> Part of the [FSDP2 profiler guides](README.md). Read the shared
> [ground rules](README.md#ground-rules-or-your-numbers-are-noise) first. This is
> the FSDP2-specific companion to the cross-example
> [`../../common/profilers/rocprofv3.md`](../../common/profilers/rocprofv3.md).

`rocprofv3` is the in-box ROCm profiler. It traces the whole process at the
**GPU-kernel** level — no framework knowledge, no code change — so it sees the
RCCL collectives exactly as they run on the hardware. Because it traces the whole
process, bound the run with a short `--warmup`/`--iters` instead of `--profile`.

> **RCCL 2.27 fuses the collectives.** On this stack every collective — the
> parameter all-gather **and** the gradient reduce-scatter — runs as one fused
> device kernel, `ncclDevKernel_Generic_2`. You will **not** see separate
> `ncclDevKernel_AllGather_*` / `…_ReduceScatter_*` rows (older RCCL builds show
> those). rocprofv3 therefore gives you the *combined* RCCL device time; to split
> all-gather vs. reduce-scatter, read the framework rows in
> [torch.profiler](torch-profiler.md#1-baseline-the-per-op--rccl-table) or the
> per-collective [TraceLens report](tracelens.md#4-the-rccl-report-per-collective-bandwidth--skew).
> The FSDP2 shape is still unmistakable: a large RCCL kernel and **no** `AllReduce`
> (that would mean DDP, not FSDP2).

## 1. Per-kernel totals (`--kernel-trace --stats`)

```bash
salloc --gpus=4 --ntasks=1 --exclusive --time=00:30:00
module load rocm openmpi pytorch
export UPSTREAM=~/pytorch_examples/distributed/FSDP2

rocprofv3 --kernel-trace --stats --truncate-kernels --output-format csv \
  -d prof_kern -- \
  torchrun --standalone --nproc_per_node=4 ../benchmarks/fsdp2_bench.py \
    --warmup 3 --iters 10
```

`--stats` writes a `*_kernel_stats.csv` ranking kernels by total time. Filter it
to the collectives to read the RCCL share directly (the kernel is
`ncclDevKernel_Generic_*` on this stack, so grep `nccl`):

```bash
grep -i 'nccl' prof_kern/*_kernel_stats.csv
```

Actual rank-0 roll-up (MI300A, 4 GPUs, 16 layers/dim 1024, fp32; `Percentage` is
the CSV column). GEMM is split across many auto-tuned `Cijk_*` Tensile kernels, so
it is grouped here:

| kernel | calls | total ms | % GPU |
|--------|------:|---------:|------:|
| `ncclDevKernel_Generic_2` (**all RCCL**, fused) | 667 | **1370.7** | **44.6** |
| GEMM (sum of `Cijk_*` Tensile variants) | ~5000 | ~1130 | ~37 |
| `bwd_kernel_fuse` (attention backward) | 208 | 177.7 | 5.8 |
| `multi_tensor_apply_kernel` (optimizer) | 975 | 63.2 | 2.1 |
| `attn_fwd` | 208 | 61.6 | 2.0 |
| `vectorized_elementwise_kernel` | 2324 | 49.6 | 1.6 |

The single largest kernel is the fused RCCL collective at **44.6 %** — but recall
that number is **skew-inflated** (a rank waiting for the slowest peer still shows
the full kernel duration). `torch.profiler`'s framework rows split it into
all-gather (~27 %) and reduce-scatter (~6 %); [TraceLens](tracelens.md) separates
the true transport from the wait. All three tools agree there is **no** `AllReduce`
— this is genuinely sharded.

## 2. RCCL tuning, seen in the totals: `NCCL_ALGO` / `NCCL_PROTO`

RCCL reads its `NCCL_*` settings when the communicator is built (at
`init_process_group`), so set them in the environment of the whole `torchrun`.
Change **one** at a time and re-read the same CSV rows:

```bash
# baseline (Ring), then Tree
for algo in Ring Tree; do
  NCCL_ALGO=$algo rocprofv3 --kernel-trace --stats --truncate-kernels \
    --output-format csv -d prof_$algo -- \
    torchrun --standalone --nproc_per_node=4 ../benchmarks/fsdp2_bench.py --warmup 3 --iters 10
  echo "== $algo =="; grep -i 'nccl' prof_$algo/*_kernel_stats.csv
done
```

**Expect:** on RCCL 2.27 the kernel **name stays `ncclDevKernel_Generic_2`**
regardless of algorithm (the algo is not encoded in the fused name), so watch the
**total time / percentage** of that row, not the name. `Ring` maximizes bandwidth
for the large all-gather buffers; `Tree` cuts latency and often wins once a
collective crosses APUs (12-/24-GPU CPX). Repeat for `NCCL_PROTO=Simple|LL|LL128`
(`LL128` is usually the medium-message sweet spot). These are the same levers as
[`../README_rccl_optimization.md` §2](../README_rccl_optimization.md#section-2--tune-the-rccl-transport--algorithm),
here measured as device-kernel time.

| setting | `ncclDevKernel_Generic_2` total ms | % GPU |
|---------|-----------------------------------:|------:|
| `NCCL_ALGO=Ring` (default) | 1370.7 | 44.6 |
| `NCCL_ALGO=Tree` | *(measure)* | *(measure)* |
| `NCCL_PROTO=LL128` | *(measure)* | *(measure)* |

> Confirm the algorithm actually changed with `NCCL_DEBUG=INFO` (grep the log for
> `Channel`/`via` lines) — the fused kernel name won't tell you.

> **Single-node caveat.** On one MI300A the on-package fabric is nearly free, so
> `NCCL_ALGO`/`NCCL_PROTO` barely move the totals. The lever earns its keep across
> APUs/nodes — rerun on the 12-/24-GPU CPX partition to see a real difference.

## 3. Timeline: `--sys-trace` → Perfetto

To *see* the collectives interleave with compute (not just their totals), emit a
Perfetto trace. `--sys-trace` captures kernels, memory copies, HIP, **and** the
RCCL API:

```bash
# keep the window SHORT — a full sys-trace is ~100 MB/rank and is hard to load
rocprofv3 --sys-trace --output-format pftrace -d prof_tl -- \
  torchrun --standalone --nproc_per_node=2 ../benchmarks/fsdp2_bench.py --warmup 1 --iters 2
```

Open the `.pftrace` at <https://ui.perfetto.dev>, press **`Expand all`** (`>`
command palette), type `ncclDevKernel_Generic` in the search box, Enter to jump,
`f` to zoom. You will see the fused `ncclDevKernel_Generic_2` block on a GPU
`STREAM` queue and whether a **GEMM runs concurrently** on another queue (good —
the prefetch is overlapping) or the queue is **empty during the gather** (bad —
serialized; turn on prefetching, see [`rocprofiler-systems.md`](rocprofiler-systems.md)).
`--sys-trace` also adds the `ALLOCATE/COPY BYTES` and `SCRATCH MEMORY` counter
rows visible below.

![FSDP2 rocprofv3 --sys-trace timeline: the fused ncclDevKernel_Generic_2 collective on STREAM 0 with GPU memory-copy counter rows](figs/fsdp2_rocprofv3_timeline.png)

> A full `--sys-trace` of a 4-GPU/6-iter run is ~97 MB/rank, which crashes a
> headless Chrome tab; the committed figure uses the small
> `--nproc_per_node=2 --iters 2` trace above (~58 MB). For interactive viewing in
> a desktop browser the large trace is fine.

## 4. Bandwidth counters (optional)

```bash
printf 'pmc: FETCH_SIZE WRITE_SIZE L2CacheHit VALUBusy\n' > counters.txt
rocprofv3 -i counters.txt --output-format csv -d prof_pmc -- \
  torchrun --standalone --nproc_per_node=4 ../benchmarks/fsdp2_bench.py --warmup 3 --iters 10
```

`FETCH_SIZE`+`WRITE_SIZE` ÷ kernel time = achieved HBM bandwidth per kernel — or
let [rocprof-compute](../../common/profilers/rocprof-compute.md) do the roofline
math for the GEMMs.

## 5. Capturing the screenshots

The Perfetto timeline PNG is captured headlessly (no display) with
[`screenshot_perfetto.py`](screenshot_perfetto.py); the full pipeline (trace +
PNG for baseline and one tuned variant) is
[`submit_timeline_traces.sbatch`](submit_timeline_traces.sbatch):

```bash
module load google-chrome/stable
export CHROME_BIN=$(command -v google-chrome)
# rocprofv3 writes the pftrace into a host subdir; pick one worker rank's file
python screenshot_perfetto.py prof_tl/*/*_results.pftrace \
  figs/fsdp2_rocprofv3_timeline.png "FSDP2 rocprofv3" 20000 ncclDevKernel_Generic 5
```

The `.pftrace` files are **not committed** (large binaries) — regenerate them with
the commands above or the batch job.

## 6. Participant exercises

1. **Read the split.** From the `--stats` CSV, take the `ncclDevKernel_Generic_2`
   `Percentage`. *What is the combined RCCL share at 4 GPUs (≈44.6 % here)?* Then
   open the `torch.profiler` table — which framework rows split it into all-gather
   vs. reduce-scatter, and do the tools agree on the total?
2. **Change the algorithm.** Run `NCCL_ALGO=Ring` vs `Tree` (§2). *The kernel name
   stays `Generic_2` — did the **total ms** move? Confirm the algorithm actually
   changed with `NCCL_DEBUG=INFO`. Why is the difference small on one APU?*
3. **Find the overlap (or the stall).** In the `--sys-trace` timeline (§3), search
   `ncclDevKernel_Generic` and look at the neighboring `STREAM` queue. *Is a GEMM
   running concurrently, or is the queue idle during the collective?* Then compare
   to the [rocprofiler-systems](rocprofiler-systems.md) host+GPU view.
4. **Spot the backward all-gather.** In the `torch.profiler` table, note that
   `nccl:_all_gather_base` has ~2× the calls of `nccl:_reduce_scatter_base` (the
   parameters are re-gathered for the backward). Apply `reshard_after_forward=False`
   by hand and recount. *By how much did the all-gather call count drop?*

## See also

- [torch.profiler](torch-profiler.md) — the framework-native equivalent table
- [rocprofiler-systems](rocprofiler-systems.md) — full host+GPU timeline with prefetch overlap
- [TraceLens](tracelens.md) — turn this trace into an RCCL bandwidth/skew table
- [rocprof-compute](../../common/profilers/rocprof-compute.md) — roofline of the GEMM kernels
