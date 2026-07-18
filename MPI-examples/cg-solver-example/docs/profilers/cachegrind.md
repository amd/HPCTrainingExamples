# Valgrind cachegrind — CG-CPU (deterministic cache model)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.

Where [perf](perf.md) reads noisy hardware counters, **cachegrind** *simulates* the
cache hierarchy, so its miss counts are **deterministic and repeatable** — ideal
for comparing two SpMV implementations without run-to-run counter jitter. `valgrind`
is installed on the MI300A **compute** nodes (`/usr/bin/valgrind`; not on the login
node). Expect a ~20–50× slowdown, so use a fixed, small matrix.

## 1. Simulate and annotate

```bash
cd CG-CPU && make CXXFLAGS="-O2 -g -std=c++17"   # -g (and -O2) for clean annotation
valgrind --tool=cachegrind --cache-sim=yes \
  --cachegrind-out-file=cg.cachegrind.out \
  ./cg_cpu src/Dubcova2.pm 12345
cg_annotate cg.cachegrind.out            # file:function miss breakdown
cg_annotate cg.cachegrind.out src/cg.cpp # annotate a source file line-by-line
```

Verified on MI300A:

```
D refs:      46,177,102  (32,130,783 rd + 14,046,319 wr)
D1  misses:     653,773        D1  miss rate: 1.4%
LLd misses:     136,346        LLd miss rate: 0.3%
```

`cg_annotate` breaks the D1/LL misses down **per function and per source line**, so
you can see the misses land in the CSR SpMV row-loop and the column-index-driven
gather of `x[col[j]]` — the classic irregular-access hotspot of sparse mat-vec.

> Because CG reads a gzipped matrix, `gzgets`/decompression can dominate a tiny run;
> fix the seed and use a larger matrix, or focus on the solve-loop functions.
> `--branch-sim=yes` adds branch-misprediction stats.

## 2. Viewing the results remotely

`cg_annotate` output is text (per-line miss columns) and works over plain SSH. For a
graphical call/cost browser, open the output in **KCachegrind**:

```bash
kcachegrind cg.cachegrind.out
```

Launch inside an AAC6 graphical session:

- `man aac6_vnc` — TurboVNC desktop, then `kcachegrind`
- `man aac6_novnc` — the same desktop in your local browser
- `man aac6_x11` — `ssh -X` then `kcachegrind` (single window)

## See also

- [Linux perf](perf.md) — measured (non-simulated) counters + hotspots
- [AMD uProf](uprof.md) — hardware hotspots + memory analysis
