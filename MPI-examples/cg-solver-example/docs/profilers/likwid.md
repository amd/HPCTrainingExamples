# likwid — CG-CPU (CPU roofline)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.

likwid builds a CPU roofline from two halves: **`likwid-bench`** microbenchmarks
measure the machine's *ceilings* (peak FLOP/s and the L1/L2/L3/DRAM bandwidths), and
**`likwid-perfctr`** reads hardware counters to place the *application point* (its
achieved FLOP/s and arithmetic intensity). With **likwid ≥ 5.5.2** (which adds
MI300A support) both halves work on the APU core for compute and the cache levels —
only the DRAM (UMC uncore) counters are still unreadable on this node.

## 1. The intended workflow (per rank)

```bash
module load rocm/7.2.4 likwid openmpi   # `likwid` resolves to 5.5.2 (MI300A-aware)
cd CG-CPU && make
likwid-mpirun -mpi openmpi -np 4 -g FLOPS_DP ./cg_cpu src/Dubcova2.pm  # DP FLOP/s
likwid-mpirun -mpi openmpi -np 4 -g L3       ./cg_cpu src/Dubcova2.pm  # L3 bandwidth
likwid-mpirun -mpi openmpi -np 4 -g MEM_DP   ./cg_cpu src/Dubcova2.pm  # DRAM (see caveat)
```

`FLOPS_DP` gives the app's DP FLOP/s and the `L2`/`L3` groups give the byte traffic at
each cache level directly (AI = FLOP ÷ byte). To isolate the SpMV from the dot
products, add the likwid **Marker API** (`likwid_markerStartRegion("spmv")` /
`...StopRegion`) and build with `-DLIKWID_PERFMON -llikwid`.

> **⚠ MI300A caveat (measured, likwid 5.5.2).** The 5.5.2 build **correctly
> identifies the core** — `likwid-perfctr -i` reports `CPU type: AMD K19 (Zen4)`,
> family 25, model 144 (`0x90`), `zen4` — a fix over 5.5.1, which reported
> `CPU type: nil`. Core-PMC groups now read via **perf_event** and work even though
> this node has **no `/dev/cpu/*/msr` device** (so `-M 0`/`-M 1` MSR access would
> otherwise fail): `FLOPS_DP`, `L2`, and `L3` all return real numbers (see §2a).
> What still **does not** work is the **`MEM`/`MEM_DP` DRAM group**: likwid knows the
> APU's 12 UMC channels (`UMC0C0…UMC11C1`) but every one reads `-`, so DRAM
> bandwidth comes back **0** — the kernel on this node exposes no `amd_umc`/`amd_df`
> uncore PMU and there is no MSR device to reach them. So we still take **DRAM byte
> traffic from [cachegrind](cachegrind.md)** and read everything else natively.
> (Two cosmetic snags: likwid prints a harmless *"Cannot open ROCm HIP library to
> fill GPU topology"* probe error, and the reported `DP [MFLOP/s]` is diluted over
> the whole-process RDTSC runtime — use the **FLOP count over the solve time** for
> the point.)

## 2. Ceilings with `likwid-bench`

`likwid-bench` runs hand-tuned assembly kernels and times them, so it needs **no**
PMU support at all. It gives every ceiling of the roofline for one core
([`measure_likwid_bench.sh`](../../CG-CPU/measure_likwid_bench.sh)):

```bash
module load rocm/7.2.4 likwid
# Compute peak (double precision, 1 thread, L1-resident working set):
likwid-bench -t peakflops_avx512_fma -w S0:24kB:1     # ~59 GFLOP/s  (vector+FMA)
likwid-bench -t peakflops             -w S0:24kB:1     # ~9.5 GFLOP/s (scalar)
# Read bandwidth per cache level (working set sized to sit in L1/L2/L3/DRAM):
likwid-bench -t load_avx512 -w S0:16kB:1              # L1   ~213 GB/s
likwid-bench -t load_avx512 -w S0:512kB:1             # L2   ~118 GB/s
likwid-bench -t load_avx512 -w S0:16MB:1              # L3   ~92  GB/s
likwid-bench -t load_avx512 -w S0:1GB:1               # DRAM ~22  GB/s (one core)
```

Measured on one MI300A APU core (`likwid-topology` reports 32 kB L1 / 1 MB L2 /
32 MB L3), single thread, double precision:

| Ceiling | `likwid-bench` kernel | Value (1 core) |
|---|---|---|
| Vector compute peak | `peakflops_avx512_fma` | **59.0 GFLOP/s** |
| Scalar compute peak | `peakflops` | **9.5 GFLOP/s** |
| L1 read bandwidth | `load_avx512` (16 kB) | **213 GB/s** |
| L2 read bandwidth | `load_avx512` (512 kB) | **118 GB/s** |
| L3 read bandwidth | `load_avx512` (16 MB) | **92 GB/s** |
| DRAM read bandwidth | `load_avx512` (1 GB) | **22 GB/s** |

## 2a. Application point with `likwid-perfctr` (native)

With 5.5.2 the point comes straight from likwid counters. Run each group on the
solver (fixed seed for reproducibility) and read the highlighted lines:

```bash
module load rocm/7.2.4 likwid
likwid-perfctr -C 0 -g FLOPS_DP ./cg_cpu src/Dubcova2.pm 12345   # compute
likwid-perfctr -C 0 -g L2       ./cg_cpu src/Dubcova2.pm 12345   # L2 traffic
likwid-perfctr -C 0 -g L3       ./cg_cpu src/Dubcova2.pm 12345   # L3 traffic
```

Measured on one MI300A APU core (`ppac-pl1-s24-16`, Dubcova2, 172 iters, seed 12345):

| Group | likwid counter / metric | Value | Used for |
|---|---|---|---|
| `FLOPS_DP` | `RETIRED_SSE_AVX_FLOPS_ALL` | **577,829,503 FLOP** (≈0.578 GFLOP) | compute rate |
| `FLOPS_DP` | solve time (program output) | **0.163 s** → **3.5 GFLOP/s** | point height |
| `L2` | L2 data volume | **5.96 GB** → AI ≈ **0.097 FLOP/B** | point vs L2 |
| `L3` | L3 data volume | **4.89 GB** → AI ≈ **0.118 FLOP/B** | point vs L3 |
| `MEM_DP` | UMC data volume | **0** (unreadable — see caveat) | DRAM ← cachegrind |

> Use the **FLOP count over the 0.163 s solve** (`0.578e9 / 0.163 ≈ 3.5 GFLOP/s`),
> not likwid's printed `DP [MFLOP/s]`, which averages over the whole process runtime
> (startup + matrix generation) and so under-reports the solve rate.

The `L1` and `DRAM` byte counts still come from [cachegrind](cachegrind.md) (the UMC
group reads zero here). Render the whole picture with
[`render_likwid_roofline.py`](../../CG-CPU/render_likwid_roofline.py) (one batch job,
[`submit_likwid_roofline.sbatch`](../../CG-CPU/submit_likwid_roofline.sbatch)):

```bash
cd CG-CPU
sbatch submit_likwid_roofline.sbatch   # writes docs/profilers/figs/cg_likwid_roofline_cpu_n1.png
```

![likwid roofline for cg_cpu on one MI300A APU core: L1/L2/L3/DRAM bandwidth diagonals, vector and scalar compute peaks, and the cg_cpu point plotted against L1, L2, L3 and DRAM traffic. Circles are native likwid-perfctr counters (L2, L3); diamonds are cachegrind (L1, DRAM)](figs/cg_likwid_roofline_cpu_n1.png)

### Reading this roofline

The four `cg_cpu` dots are the **same** run (≈3.5 GFLOP/s) plotted against the bytes
it moves at four different levels — exactly like the *Memory region* dropdown in the
[GPU roofline](roofline-extractor.md). Circles (**L2, L3**) are read natively by
`likwid-perfctr`; diamonds (**L1, DRAM**) come from cachegrind:

- **vs DRAM** (red, AI ≈ 13.7 FLOP/B): far to the right, well below every roof.
  Dubcova2 (~13 MB of CSR) lives in the 32 MB L3 / large Infinity Cache, so almost
  nothing reaches DRAM — CG is **not** DRAM-bound on the APU.
- **vs L2 / L3** (green/orange, AI ≈ 0.097 / 0.118 FLOP/B): the native likwid points.
  Both sit a factor ~3 **below** their bandwidth diagonals (L2 118, L3 92 GB/s), so
  CG is not cache-bandwidth-bound either.
- **vs L1** (blue, AI ≈ 0.036 FLOP/B): the tightest one. It sits **just below the
  L1 bandwidth diagonal** — CG reaches only ~45 % of the achievable L1 read
  bandwidth. It is **L1-gather-latency-bound**, not L1-bandwidth-bound: the
  irregular `x[A.col_idx[j]]` gather that [cachegrind](cachegrind.md) pins on one
  line can't stream L1 at full width.
- The binding **compute** ceiling is the **scalar 9.5 GF/s** roof, not the 59 GF/s
  vector peak — the gather defeats vectorization, so the 6× headroom above is
  unreachable.

> **likwid-bench vs uProf's DRAM roof.** The [uProf roofline](uprof.md) draws the
> **full-socket theoretical** DRAM roof (1700 GB/s). likwid-bench measures what a
> **single core** can actually pull — **~22 GB/s**, ~77× less — because one Zen4
> core cannot saturate the APU's HBM. For a *single-threaded* kernel the likwid
> number is the honest ceiling; the 1700 GB/s roof only matters once all 24 cores of
> a socket pull together.

## 3. GPU counters (rocmon): sees the GPU, can't read it here

likwid 5.5.2 is built **GPU-enabled** (`rocmon` for ROCm 7.2.4): `likwid-perfctr` has
`-I <gpus>` / `-R <rocmgroup>` and carries AMD-GPU groups
(`VALU, WAVE, MEM, SALU, SFETCH, GDS, PCI, POWER, STALLED, UTIL`). With the ROCm HIP
library on the path it can **detect** the MI300A GPU:

```bash
module load rocm/7.2.4 likwid
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH   # else "Cannot open ROCm HIP library"
likwid-perfctr -i        # -> "ROCMON GPU 0 compute capability: 9.4 … amd_gpu"
```

But **reading** any GPU counter still fails on this chip:

```text
Initialize(), input metric 'SQ_INSTS_VALU' is not supported on this hardware: gfx942
Error: Context Create failed
```

> **Why (measured).** `rocmon` drives the **legacy rocprofiler v1** API
> (`librocprofiler64.so.1`). On MI300A (gfx942) under ROCm 7.x, v1 is end-of-life:
> even though `gfx_metrics.xml` still lists `SQ_INSTS_VALU`, v1 cannot *create a
> profiling context* on gfx942 — counter collection moved to rocprofiler-sdk /
> `rocprofv3`, which the likwid `rocmon` backend does not use. (Two lesser snags:
> Slurm presets `ROCR_VISIBLE_DEVICES`, so `unset` it or `-I 0` errors; and a few
> shipped groups — e.g. `MEM` — reference counters like `TA_TA_BUSY` absent on
> gfx942.) **Net: likwid sees the GPU but cannot read its counters here** — this is
> unchanged from 5.5.1. For GPU counters/roofline use
> [rocprofv3](rocprofv3.md), [rocprof-compute](rocprof-compute.md), or
> [roofline-extractor](roofline-extractor.md).

## 4. Participant exercises

Build first (`cd CG-CPU && make`) and work these on an MI300A node
(`salloc -p PPAC_MI300A_SPX -N1 -c8`), `module load rocm/7.2.4 likwid` (5.5.2).

1. **Confirm the core is now recognized, and read a counter.** Run
   `likwid-perfctr -i | head -20` and confirm `CPU type: AMD K19 (Zen4)`,
   `zen4` (5.5.1 said `nil`). Then run `likwid-perfctr -C 0 -g FLOPS_DP -- sleep 0.2`
   and confirm `RETIRED_SSE_AVX_FLOPS_ALL` is populated — even though
   `ls /dev/cpu/0/msr` shows *no such file*. *Which access mode makes this work
   without an MSR device?* (Hint: perf_event, `perf_event_paranoid` is `-1` on the
   compute node.)

2. **Find the group that still fails, and say why.** Run
   `likwid-perfctr -C 0 -g MEM -- sleep 0.2`. *What do the `UMC*C*` counters read,
   and what is the resulting "Memory bandwidth"?* Contrast with `-g L3`, which works.
   Explain the difference in terms of **core PMC vs UMC uncore** counters and which
   PMUs `ls /sys/bus/event_source/devices/` actually exposes on the node.

3. **Measure the bandwidth hierarchy yourself.** Sweep `load_avx512` across working
   sets that fit L1 (16 kB), L2 (512 kB), L3 (16 MB) and exceed all caches (1 GB).
   *Do the four bandwidths fall the way the table shows (213 → 118 → 92 → 22 GB/s)?*
   Why is the L1↔DRAM ratio here (~10×) so much smaller than on a GPU (HBM vs LDS)?

4. **Place the native point.** Run `FLOPS_DP`, `L2` and `L3` on the solver
   (`./cg_cpu src/Dubcova2.pm 12345`). From the FLOP count over the solve time and the
   L2/L3 data volumes, recompute `cg_cpu`'s throughput and its AI vs L2 and L3. Add
   the cachegrind L1/DRAM points. *Which dot lands nearest a roof, and what does its
   distance below that roof tell you* (bandwidth-bound vs latency-bound)?

5. **All cores vs one core.** Re-run `likwid-bench -t load_avx512 -w S0:2GB:24`
   (24 threads) and compare the aggregate DRAM bandwidth to the single-core 22 GB/s
   and to uProf's 1700 GB/s socket roof. *How many cores does it take to approach the
   socket roof?* Relate this to why a single-rank CG never sees the HBM roof.

6. **CPU vs GPU limiter.** Put this figure next to the [GPU roofline]
   (roofline-extractor.md) of the same solver. *Why is CG L1-latency/scalar-bound on
   the APU CPU core but HBM-bandwidth-bound on the GPU?* Answer in one sentence tying
   it to cache residency and arithmetic intensity.

## Viewing the results

`likwid-bench`/`likwid-perfctr` output is text (per-region tables); no graphical
viewer is needed. The roofline PNG above is produced headlessly by
`render_likwid_roofline.py`; redirect the raw benchmark output to a file to compare
runs.

## See also

- [AMD uProf](uprof.md) — the companion CPU roofline (uProf's own plotter; full-socket
  DRAM roof)
- [Linux perf](perf.md) — the always-available CPU counter fallback on MI300A
- [Valgrind cachegrind](cachegrind.md) — deterministic byte traffic behind the AI
- [roofline-extractor](roofline-extractor.md) — the **GPU** roofline, where the same
  solver is HBM-bandwidth-bound
