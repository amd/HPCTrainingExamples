# Linux perf — CG-CPU (always-available CPU baseline)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.
> See [`perf-security.md`](perf-security.md) for the `perf_event_paranoid` model.

`perf` needs no modules and no special build — the zero-friction CPU baseline, and
the natural fallback where [likwid](likwid.md)'s counters are unsupported on MI300A.
The MI300A **compute** nodes run with `perf_event_paranoid = -1`, so hardware
counters are fully available (the **login** node is restricted — run perf inside an
allocation).

## 1. Counter summary + hotspots

```bash
cd CG-CPU && make CXXFLAGS="-O3 -g -std=c++17"   # -g for symbol resolution
# Per-rank counter summary (IPC + cache behaviour), one file per rank:
mpirun -n 4 bash -c 'perf stat -o perf_r${OMPI_COMM_WORLD_RANK}.txt \
  -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
  ./cg_cpu src/Dubcova2.pm 12345'
# Hotspot sampling → where the cycles go:
mpirun -n 1 perf record -g -o perf.data ./cg_cpu src/Dubcova2.pm 12345
perf report -i perf.data        # interactive TUI; or `perf annotate` for source+asm
```

Verified single-rank on MI300A (`perf stat`):

```
    726,966,836  cycles
    602,835,327  instructions            #  0.83  insn per cycle
     18,270,624  cache-references
      2,410,388  cache-misses            #  13.19% of all cache refs
    205,921,430  L1-dcache-loads
      4,193,649  L1-dcache-load-misses   #   2.04% of all L1-dcache accesses
```

The low IPC (0.83) and the non-trivial last-level cache-miss rate (~13%) are the
CPU signature of the memory-bound SpMV — the CPU analog of the GPU HBM-bound
roofline. `perf record`/`perf report` then attribute cycles to the SpMV row-loop
and the CSR index gather.

## 2. Viewing the results remotely

`perf report`/`perf annotate` are terminal (TUI) tools — they work over plain SSH,
no graphical session needed. For a **flame graph** of the hotspots:

```bash
perf script -i perf.data | stackcollapse-perf.pl | flamegraph.pl > cg_cpu_flame.svg
```

View the SVG in a graphical session:

- `man aac6_vnc` — TurboVNC desktop, open `cg_cpu_flame.svg` in a browser
- `man aac6_novnc` — the same desktop in your local browser
- `man aac6_x11` — `ssh -X` and open a single browser/image window
- or just `scp` the small SVG to your workstation.

## See also

- [Valgrind cachegrind](cachegrind.md) — deterministic (simulated) cache misses
- [AMD uProf](uprof.md) — hotspots + memory/bandwidth analysis with a GUI
- [perf-security.md](perf-security.md) — why perf is restricted on the login node
