# TAU — CG-GPU / CG-CPU (the MPI communication view)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.

TAU intercepts MPI (and, with `-rocm`, ROCm) so it directly measures the
communication the ROCm GPU tools miss — per-call MPI time and a **communication
matrix** across ranks.

## 1. Collect a profile

```bash
module load tau rocm openmpi
cd CG-GPU && make
# GPU solver: MPI + ROCm
CG_SEED=12345 mpirun -n 4 ./gpu_bind.sh \
  tau_exec -T MPI,ROCM -rocm ./cg_gpu src/Dubcova2.pm rccl
# CPU reference: MPI only
mpirun -n 4 tau_exec -T MPI ./cg_cpu src/Dubcova2.pm

pprof            # text summary
```

TAU reports time in `MPI_Allreduce` (the dot-product reduction → `g_allreduce_time`),
`MPI_Isend`/`MPI_Irecv`/`MPI_Waitall` or `MPI_Alltoallv` (the halo exchange →
`g_halo_time`), plus GPU kernel time — the same split the solver prints, now
attributed per MPI call.

> For this small test matrix `MPI_Init` dominates the profile; use a larger matrix
> for a meaningful loop breakdown. TAU may return a nonzero exit at teardown after
> ROCm tracking — the `profile.*` files are still written and readable.

## 2. Viewing the results remotely

- **`pprof`** — text summary; works over plain SSH.
- **`paraprof`** — the Java GUI with the per-call bar charts and the **communication
  matrix**. Launch it inside a graphical session:
  - `man aac6_vnc` — TurboVNC desktop, then `paraprof`
  - `man aac6_novnc` — the same desktop in your local browser
  - `man aac6_x11` — `ssh -X` then `paraprof` (single window)

> **JRE note (this cluster).** `paraprof` (and `jumpshot`) are Java Swing apps and
> need a JRE, which is **not currently installed** on the AAC6 nodes. Until a
> `java`/JDK module is available, use `pprof` for the text profile, or copy the
> `profile.*` files to a workstation that has ParaProf installed.

## See also

- [HPCToolkit](hpctoolkit.md) — call-path sampling alternative for MPI+GPU
- [rocprofv3](rocprofv3.md) — GPU kernels/transports (no MPI)
