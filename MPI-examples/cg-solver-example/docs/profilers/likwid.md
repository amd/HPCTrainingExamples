# likwid — CG-CPU (CPU roofline)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.

On a supported CPU, likwid gives hardware-counter bandwidth/FLOP/s per rank.

## 1. Bandwidth and FLOP/s

```bash
module load likwid openmpi
cd CG-CPU && make
likwid-mpirun -mpi openmpi -np 4 -g MEM_DP  ./cg_cpu src/Dubcova2.pm   # DRAM bandwidth
likwid-mpirun -mpi openmpi -np 4 -g FLOPS_DP ./cg_cpu src/Dubcova2.pm  # DP MFLOP/s
```

Compare the summed DRAM bandwidth to the node's STREAM peak: a memory-bound CG
should land near it. To isolate the SpMV from the dot products, add the likwid
**Marker API** (`likwid_markerStartRegion("spmv")` / `...StopRegion`) and build with
`-DLIKWID_PERFMON -llikwid`.

> **⚠ MI300A caveat (measured).** On the MI300A APU node, `likwid-perfctr` (v5.5.1)
> reports *"Unsupported AMD Zen Processor / K19"* and cannot initialize its event
> groups — the integrated Zen4 CPU is not in this build's perfmon map, so
> `MEM_DP`/`FLOPS_DP` are unavailable there. `likwid-topology` still works. **For
> CPU counters/hotspots on MI300A, use [Linux perf](perf.md), [AMD uProf](uprof.md),
> or [rocprof-compute](rocprof-compute.md) for the GPU roofline instead.** likwid
> remains useful on conventional EPYC login/compute nodes.

## Viewing the results

likwid output is text (per-region tables); no graphical viewer is needed. Redirect
to a file per group and compare across runs.

## See also

- [Linux perf](perf.md) — the always-available CPU counter fallback on MI300A
- [AMD uProf](uprof.md) — CPU hotspots + memory analysis with a GUI
