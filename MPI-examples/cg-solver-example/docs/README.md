# Communication & Performance Tutorial — Distributed CG on AMD GPUs

A step-by-step tutorial built around the two examples in this repository:

- **`CG-CPU/`** — a distributed, CPU-only Conjugate Gradient (CG) solver (reference implementation).
- **`CG-GPU/`** — the same solver ported to AMD GPUs (rocSPARSE / rocBLAS / RCCL) with **five interchangeable
  communication variants** for the halo exchange.

The goal is not just to run the examples, but to walk the **path to a trustworthy, optimized performance
measurement** so you can fairly compare communication approaches on AMD Instinct MI300A hardware.

## Who this is for

You are comfortable with C++/MPI and have access to an MI300A node (or similar CDNA3 APU) with ROCm and a
GPU-Aware MPI. You want to understand *how* to measure GPU communication cost correctly and *which* transport
to pick.

## The path (read in order)

| # | Chapter | What you get |
|---|---------|--------------|
| 0 | [Introduction & setup](00-introduction.md) | The problem, the two examples, hardware/software prerequisites |
| 1 | [The CPU reference](01-cpu-reference.md) | Build/run `CG-CPU`, understand the algorithm and distributed SpMV |
| 2 | [Porting to the GPU](02-gpu-port.md) | Build/run `CG-GPU`, what changed (rocSPARSE/rocBLAS/RCCL/GPU-Aware MPI) |
| 3 | [Measuring performance correctly](03-correct-measurement.md) | **The core chapter:** affinity, reproducibility, warm-up, repeats, isolating comm vs compute |
| 4 | [Comparing communication variants](04-communication-variants.md) | The five transports, a fair comparison, and how to read the results |
| 5 | [Tuning the copy engines (SDMA vs blit)](05-sdma-vs-blit.md) | `HSA_ENABLE_SDMA` / `_GANG`, when each wins |
| 6 | [ROCm version & the SpMV API](06-rocm-version-and-spmv.md) | Version sweeps, the 7.x compute regression, `rocsparse_v2_spmv`, OpenMPI vs cray-mpich |
| 7 | [The optimized configuration](07-optimization-path.md) | Putting it together: a recommended launch, a checklist, and where to go next |
| 8 | [Profiling communication & compute](08-profiling.md) | The system profilers — rocprofv3 (incl. ATT), rocprof-compute, rocprofiler-systems, TAU, HPCToolkit, likwid, AMD uProf, IntelliKit, roofline-extractor, rocBudAI, Linux perf, cachegrind — and perf_events security |
| 9 | [perf_events security — implementation & demo](09-perf-security-demo.md) | The `perf_event_paranoid` ladder, `CAP_PERFMON`, resource limits; a no-root demo on `CG-CPU` (kernel-level, ROCm-independent) |

## TL;DR — the recommended optimized measurement

If you only remember one thing: **fix affinity and the RHS seed first, then take the minimum of several warm
runs, and always split the reported time into communication vs. compute.** Everything else (which transport,
which copy engine, which ROCm version) is a second-order comparison that is only meaningful once the
measurement itself is under control.

```bash
cd CG-GPU
make
# reproducible, affinity-bound, comm/compute split reported:
CG_SEED=12345 mpirun -n 4 ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm isend
```

The two in-depth results write-ups this tutorial draws on:

- [`CG-GPU/STUDY_REPORT.md`](../CG-GPU/STUDY_REPORT.md) — AAC6 / MI300A Ubuntu, OpenMPI, ROCm 6.4.1–7.13.
- [`CG-GPU/STUDY_REPORT_PrgEnv-amd.md`](../CG-GPU/STUDY_REPORT_PrgEnv-amd.md) — HPE Cray EX, cray-mpich, ROCm 7.0.3.
