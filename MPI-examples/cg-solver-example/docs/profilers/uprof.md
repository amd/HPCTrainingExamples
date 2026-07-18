# AMD uProf — CG-CPU (CPU hotspots + memory)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.

AMD uProf **does** work on the MI300A node (unlike [likwid](likwid.md)). Profile it
**per rank under `mpirun`** (so uProf runs on the compute node where the ranks
execute), each writing its own directory, then report from that **directory**.

## 1. Collect and report

```bash
export PATH=$PATH:/nfsapps/ubuntu-24.04-nightlies/opt/AMDuProf_5.3-518/bin
cd CG-CPU && make
# Time-based (hotspot) profiling, one uProf per rank:
mpirun --oversubscribe -n 4 bash -c \
  'AMDuProfCLI collect --config tbp -o uprof_r${OMPI_COMM_WORLD_RANK} ./cg_cpu src/Dubcova2.pm'
# Report from the collection DIRECTORY (not the session.uprof file):
AMDuProfCLI report -i uprof_r0        # writes uprof_r0/report.csv
```

Measured on MI300A, `report.csv` lists the hottest functions — for this solver the
top entry is the CSR SpMV (`spmv(double, ParMat&, ...)`), then `main` and the
`std::map` column-index lookups. The memory/bandwidth analysis config (see
`AMDuProfCLI collect --help`) adds DRAM traffic and cache behaviour — the CPU-side
analog of the [rocprof-compute](rocprof-compute.md) roofline.

> Requires `perf_event_paranoid` low enough to sample; the MI300A compute nodes are
> configured permissively (see [perf-security.md](perf-security.md)).

## 2. Viewing the results remotely

- `report.csv` is text — inspect with `column -s, -t uprof_r0/report.csv` or a
  spreadsheet.
- The **AMDuProf GUI** imports a collection directory for graphical hotspot/flame
  and memory views. Launch it inside a graphical session:
  - `man aac6_vnc` — TurboVNC desktop, then `AMDuProf`
  - `man aac6_novnc` — the same desktop in your local browser
  - `man aac6_x11` — `ssh -X` then `AMDuProf` (single window)

## See also

- [Linux perf](perf.md) — lighter, always-available hotspots
- [Valgrind cachegrind](cachegrind.md) — deterministic per-line cache misses
