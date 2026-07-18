# Distributed GPU Conjugate Gradient — Communication Study (Cray EX / cray-mpich)

**Target:** `CG-GPU` distributed conjugate gradient solver (rocSPARSE / rocBLAS / RCCL + GPU-Aware MPI)
**Platform:** single AMD **MI300A** node, HPE Cray EX, RHEL 9 (`192C4G1H_MI300A_RHEL9`), 4 APU partitions
**Toolchain:** **PrgEnv-amd/8.7.0** (AMD clang / ROCm **7.0.3**), **cray-mpich 9.1.0** (GPU-aware via `craype-accel-amd-gfx942`)
**Matrix:** `Dubcova2.pm`, 65 536 × 65 536, SPD
**Config:** 4 MPI ranks (1 GPU/rank), relative-residual tolerance `1e-6`, `CG_SEED=12345`

> This is the companion to `STUDY_REPORT.md` (AAC6 / MI300A Ubuntu, ROCm 6.4.1–7.2.4, **OpenMPI 5.0.10**).
> Same solver, same matrix, same 4-rank configuration — but built with the **PrgEnv-amd** programming
> environment and run over **cray-mpich** with `srun`, so the two reports isolate the effect of the MPI
> stack and the Cray launcher/binding model.

---

## 1. Executive summary

The five communication variants (`staged`, `isend`, `rccl`, `alltoallv_staged`, `alltoallv`) were built with
`PrgEnv-amd/8.7.0` and benchmarked over GPU-aware cray-mpich on one MI300A node. Findings:

1. **Numerics are identical to the OpenMPI/AAC6 study.** With the fixed RHS seed all five methods produce the
   **same initial residual (196.121), 172 iterations, and final residual (1.815e-4)** — bit-for-bit the same as
   the AAC6 runs. The MPI stack and toolchain change data transport only, not the math.

2. **GPU-aware cray-mpich works out of the box and is competitive with GPU-aware OpenMPI.** With
   `MPICH_GPU_SUPPORT_ENABLED=1` and the `libmpi_gtl_hsa` GTL library (auto-linked by the Cray `CC` wrapper when
   `craype-accel-amd-gfx942` is loaded), the GPU-pointer variants (`isend`, `alltoallv`) and `rccl` deliver halo
   times of **0.010–0.016 s** — as good as or slightly better than the OpenMPI numbers in `STUDY_REPORT.md`.

3. **The CPU/GPU-affinity cliff seen on AAC6 does NOT reproduce here.** On AAC6 an un-bound launch was ~100×
   slower; on this Cray EX system an `--exclusive` `srun` allocation already gives each rank good NUMA-local
   placement, so adding explicit per-rank binding (`gpu_bind.sh`) buys only **~3–8 %** (see §3). The dominant
   factor on AAC6 (host-side CPU locality) is largely handled by the Cray/SLURM cgroup binding by default.

4. **Total solve time is compute-bound, consistent with the ROCm 7.x regression.** ROCm 7.0.3 compute
   (`isend` ≈ 0.079 s) sits squarely in the regressed 7.x band characterised in `STUDY_REPORT.md` §5
   (7.0.2 ≈ 0.057 s … 7.2.4 ≈ 0.091 s), and is ~2.6× the ROCm 6.4.1 baseline (≈ 0.030 s). Communication is a
   minority of the solve (14–62 %); the rocSPARSE/rocBLAS/launch path dominates.

5. **SDMA copy engines help the copy-heavy paths, as on AAC6.** Forcing shader "blit" copies
   (`HSA_ENABLE_SDMA=0`) is slower on every copy-bound variant; the SDMA default (`=1`) plus
   `HSA_ENABLE_SDMA_GANG=1` cuts halo time by up to **~24 %** on `isend`/`alltoallv` (see §5).

---

## 2. Methodology

**Toolchain / build.** The Makefile now auto-detects the Cray Programming Environment (`PE_ENV`) and builds
with the Cray `CC` wrapper and `-x hip`:

```bash
module swap PrgEnv-cray PrgEnv-amd/8.7.0     # AMD clang 20 / ROCm 7.0.3 (/opt/rocm-7.0.3)
module load craype-accel-amd-gfx942          # GPU-aware cray-mpich (links libmpi_gtl_hsa)
make                                          # CC -x hip --offload-arch=gfx942 ... -lrocsparse -lrocblas -lrccl -lamdhip64
```

The `CC` wrapper supplies the cray-mpich include/link flags and, with `craype-accel-amd-gfx942` loaded, the
GPU-aware GTL library; only the ROCm math libraries are added explicitly. The resulting binary links
`libmpi_amd.so.12` (cray-mpich) and `libmpi_gtl_hsa.so.0` (GPU-aware transport).

**Launch.** `srun` inside an `--exclusive`, `--gpus-per-node=4`, 4-task allocation. GPU-aware MPI requires
`MPICH_GPU_SUPPORT_ENABLED=1` at runtime. Per-rank GPU+NUMA binding uses `gpu_bind.sh` (sets
`ROCR_VISIBLE_DEVICES=<local_rank>` and `numactl --cpunodebind/--membind`). **Cray-specific caveat:** the
wrapper's `numactl` must be launched under `srun --cpu-bind=none`; otherwise srun's default per-task cpuset
restricts each rank to a subset of cores and `numactl --cpunodebind` fails with `sched_setaffinity: Invalid
argument` on the ranks whose target NUMA lies outside their cgroup — which then hangs the cray-mpich PMI
bootstrap barrier (180 s timeout) and aborts `MPI_Init`. `test_cg_gpu.sh` sets both `MPICH_GPU_SUPPORT_ENABLED`
and `--cpu-bind=none` automatically when it detects a Cray PE.

**Instrumentation** is unchanged from `STUDY_REPORT.md`: `g_halo_time` (SpMV ghost exchange) and
`g_allreduce_time` (dot-product allreduce), reduced as the max across ranks over the timed CG loop only.
Values below are the **average of 5 runs** (affinity comparison: 3 runs), fixed `CG_SEED=12345`.

Artifacts added for this environment (all in `CG-GPU/`):

| File | Purpose |
|------|---------|
| `Makefile` | Now auto-detects Cray PE and builds with `CC -x hip`; OpenMPI/`mpicxx` path preserved. |
| `submit_cg_gpu_cray.sbatch` | SLURM wrapper: loads PrgEnv-amd + `craype-accel-amd-gfx942`, sets `MPICH_GPU_SUPPORT_ENABLED`, runs `test_cg_gpu.sh`. |
| `cray_study.sh` | Study driver: N repeats/method over `srun`, verifies convergence, prints per-run + averaged timing. |
| `test_cg_gpu.sh` | Now exports `MPICH_GPU_SUPPORT_ENABLED=1` and launches `srun --cpu-bind=none` on Cray. |
| `results_cray_comm.txt`, `results_cray_unbound.txt`, `results_cray_sdma.txt` | Raw run logs backing §3–§5. |

---

## 3. Affinity (bound vs unbound)

Topology matches AAC6 exactly: **GPU _i_ ↔ NUMA node _i_**, 48 logical CPUs (24 cores + 24 SMT) per NUMA,
all GPUs XGMI 1-hop.

| NUMA | CPUs | GPU |
|------|------|-----|
| 0 | 0-23, 96-119   | GPU0 |
| 1 | 24-47, 120-143 | GPU1 |
| 2 | 48-71, 144-167 | GPU2 |
| 3 | 72-95, 168-191 | GPU3 |

Solve time (s), 4 ranks, `CG_SEED=12345`:

| method | unbound `srun --exclusive` | bound `--cpu-bind=none` + `gpu_bind.sh` | Δ |
|--------|---------------------------|------------------------------------------|---|
| staged           | 0.1795 | 0.1722 | −4.1 % |
| isend            | 0.1007 | 0.0973 | −3.4 % |
| rccl             | 0.0930 | 0.0926 | ≈ 0 |
| alltoallv_staged | 0.1368 | 0.1261 | −7.8 % |
| alltoallv        | 0.0996 | 0.0971 | −2.5 % |

**Key contrast with AAC6.** On AAC6 the un-bound path was catastrophic (~100× slower, `isend` 10.6 s → 0.05 s)
because a non-exclusive `--gpus=4 --ntasks=4` allocation handed the whole job only ~4 remote hardware threads
and `mpirun --bind-to none` let all ranks contend for them. On this Cray EX system, an **`--exclusive` `srun`
allocation already binds each rank to a distinct, largely NUMA-local core set by default**, so explicit binding
adds only a few percent. The affinity cliff is a property of the (non-exclusive, un-bound) launch setup on
AAC6, not of the solver — and the Cray SLURM defaults avoid it. Explicit `gpu_bind.sh` binding is still the
recommended launch (best and most reproducible), but it is not the make-or-break factor it was on AAC6.

---

## 4. Communication timing (bound, fixed seed = 12345, avg of 5 runs)

All methods: init residual 196.121, **172 iterations**, final residual 1.815e-4 (identical to AAC6). Times are
seconds, max across ranks, CG loop only.

| method | solve | comm total | halo exch | dot allreduce | compute | comm % |
|--------|-------|-----------|-----------|---------------|---------|--------|
| staged            | 0.1722 | 0.1064 | 0.0862 | 0.0202 | 0.0657 | 62 % |
| isend             | 0.0973 | 0.0182 | 0.0146 | 0.0036 | 0.0791 | 19 % |
| rccl              | 0.0926 | 0.0129 | 0.0099 | 0.0030 | 0.0796 | 14 % |
| alltoallv_staged  | 0.1261 | 0.0547 | 0.0439 | 0.0107 | 0.0714 | 43 % |
| alltoallv         | 0.0971 | 0.0191 | 0.0156 | 0.0035 | 0.0779 | 20 % |

### Interpretation

- **Host staging is expensive** — same qualitative result as AAC6. `staged` halo (0.086 s) vs `isend` (0.015 s)
  is a ~6× gap that is purely the D→H + H→D copies GPU-aware cray-mpich removes; `alltoallv_staged` (0.044 s)
  vs `alltoallv` (0.016 s) isolates the same PCIe round trip.
- **GPU-aware point-to-point, collective, and RCCL are comparable** (~0.010–0.016 s halo). Over intra-node
  XGMI at 4 GPUs, none has a decisive edge — differences are within run-to-run noise.
- **Compute dominates the solve here** (compute ≈ 0.066–0.080 s, i.e. 38–86 % of solve), unlike AAC6 on
  ROCm 6.4.1 where communication dominated. The cause is the ROCm 7.x compute regression (§5 of the AAC6
  report): ROCm 7.0.3 compute (≈ 0.079 s for `isend`) is ~2.6× the 6.4.1 baseline (≈ 0.030 s).
- **`dot allreduce` is cheap on cray-mpich** (0.003–0.004 s for the GPU-aware/RCCL variants), comparable to the
  best OpenMPI 7.2.4 numbers and better than OpenMPI 6.4.1.

### Cross-toolchain comparison (`isend`, 4 ranks, same node class & matrix)

| stack | ROCm | solve | comm | halo | dot allreduce | compute |
|-------|------|-------|------|------|---------------|---------|
| OpenMPI 5.0.10 (AAC6) | 6.4.1 | 0.051 | 0.028 | 0.014 | 0.014 | 0.024 |
| OpenMPI 5.0.10 (AAC6) | 7.2.4 | 0.111 | 0.020 | — | 0.003 | 0.091 |
| **cray-mpich 9.1.0 (this study)** | **7.0.3** | **0.097** | **0.018** | **0.015** | **0.004** | **0.079** |

**cray-mpich's GPU-aware communication is on par with (marginally better than) GPU-aware OpenMPI** at this
scale; the total solve difference between the toolchains is driven by the ROCm compute version, not the MPI
implementation. ROCm 7.0.3 lands, as expected, between the 7.0.2 and 7.2.4 compute points of the AAC6 sweep.

---

## 5. SDMA vs blit-kernel copy engines (avg of 5 runs)

`HSA_ENABLE_SDMA=0` forces shader "blit" copies; `=1` (default) uses the dedicated SDMA/DMA engines;
`HSA_ENABLE_SDMA_GANG=1` gangs multiple SDMA engines. Same binary across configs — only the runtime
environment varies. Halo-exchange time (s), and Δ vs blit:

| method | blit | sdma | sdma_gang | best Δ vs blit |
|--------|------|------|-----------|----------------|
| staged           | 0.0944 | 0.0890 | 0.0867 | **−8 %** |
| isend            | 0.0188 | 0.0164 | 0.0142 | **−24 %** |
| rccl             | 0.0101 | 0.0096 | 0.0099 | −5 % |
| alltoallv_staged | 0.0441 | 0.0437 | 0.0487 | −1 % (gang +10 %, noise) |
| alltoallv        | 0.0179 | 0.0145 | 0.0137 | **−23 %** |

### Findings

- **SDMA helps the copy-heavy and GPU-aware point-to-point paths**, exactly as on AAC6: `isend` (−24 %),
  `alltoallv` (−23 %), host-staged `staged` (−8 %). Blit copies consume CU/shader cycles and contend with the
  SpMV kernels; SDMA offloads them to dedicated engines.
- **Ganging adds a small extra win** on the copy-bound variants (`isend` −13 %→−24 %, `alltoallv`
  −19 %→−23 %) and is within noise for `rccl`/`alltoallv_staged`.
- The ROCm default (`HSA_ENABLE_SDMA=1`) is the right choice; forcing blit costs up to ~24 % of halo time on
  the staged and GPU-aware point-to-point paths.

---

## 6. Reproduce

```bash
cd CG-GPU

# Build + run all 5 methods on a compute node (PrgEnv-amd + cray-mpich, fixed seed):
sbatch --export=ALL,CG_SEED=12345 submit_cg_gpu_cray.sbatch

# Or interactively, inside an exclusive allocation:
module swap PrgEnv-cray PrgEnv-amd/8.7.0
module load craype-accel-amd-gfx942
make
salloc --partition=192C4G1H_MI300A_RHEL9_A1 --exclusive -N1 --ntasks=4 --gpus-per-node=4 -t 01:00:00

# Averaged communication study (5 runs/method, with affinity binding):
JOBID=<allocation> REPEATS=5 ./cray_study.sh

# Affinity comparison (unbound):
JOBID=<allocation> REPEATS=3 BIND=0 ./cray_study.sh

# SDMA copy-engine sweep:
for c in "0 unset" "1 unset" "1 1"; do read s g <<<"$c"; \
  HSA_ENABLE_SDMA=$s HSA_ENABLE_SDMA_GANG=$g JOBID=<allocation> REPEATS=5 ./cray_study.sh; done

# Single method, reproducible:
MPICH_GPU_SUPPORT_ENABLED=1 CG_SEED=12345 \
  srun -N1 -n4 --gpus-per-node=4 --cpu-bind=none ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm rccl
```

Environment: `PrgEnv-amd/8.7.0` (ROCm 7.0.3) + `craype-accel-amd-gfx942` + `cray-mpich 9.1.0`, with
`MPICH_GPU_SUPPORT_ENABLED=1` for the GPU-aware variants.
