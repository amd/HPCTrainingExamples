# HPCToolkit — CG-GPU / CG-CPU (call-path sampling)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.

HPCToolkit samples the **full call path**, showing how much time each rank spends
in MPI wait vs. compute vs. GPU kernels — ideal for spotting a load-imbalanced halo
exchange or a rank stalling in `MPI_Allreduce`.

## 1. Measure, then build the database

```bash
module load hpctoolkit rocm openmpi
cd CG-GPU && make
# GPU solver: CPU call paths + AMD GPU operations
CG_SEED=12345 mpirun -n 4 ./gpu_bind.sh \
  hpcrun -e CPUTIME -e gpu=amd -o cg_gpu.m ./cg_gpu src/Dubcova2.pm rccl
# CPU reference:
mpirun -n 4 hpcrun -e CPUTIME -o cg_cpu.m ./cg_cpu src/Dubcova2.pm

hpcstruct ./cg_gpu                       # recover program structure
hpcprof -S cg_gpu.hpcstruct cg_gpu.m -o cg_gpu.d
```

> Run `hpcstruct`/`hpcprof` on the **compute node** so their runtime libs are
> present; if `hpcprof` warns about a partial struct-file match, pass the suggested
> `-R '<build>'='<build>/.'` remap.

## 2. Viewing the database remotely

The database (`cg_gpu.d`) is opened in the graphical **hpcviewer** (call-path
profile) and **hpctraceviewer** (timeline):

```bash
hpcviewer cg_gpu.d          # call-path profile
hpcviewer cg_gpu.d          # (hpctraceviewer view for the timeline)
```

Launch inside an AAC6 graphical session:

- `man aac6_vnc` — TurboVNC desktop, then `hpcviewer`
- `man aac6_novnc` — the same desktop in your local browser
- `man aac6_x11` — `ssh -X` then `hpcviewer` (single window)

`hpcviewer` is an Eclipse/SWT app; if the cluster lacks its GUI runtime, copy the
self-contained `cg_gpu.d` database to a workstation with hpcviewer installed.

## See also

- [TAU](tau.md) — per-call MPI time + communication matrix
- [rocprofiler-systems](rocprofiler-systems.md) — GPU+host Perfetto timeline
