Distributed CG Solver Example
--

This repo contains 2 directories
1. CG-CPU - a CPU-based CG solver which serves as a reference implementation
2. CG-GPU - a GPU-based CG solver using rocm libraries and various communication methods for testing

## Tutorial

A step-by-step **Communication & Performance Tutorial** built around these two examples lives in
[`docs/`](docs/README.md). It walks from the CPU reference to the GPU port, then to the path for a trustworthy,
optimized performance measurement that fairly compares the communication approaches on AMD MI300A:

0. [Introduction & setup](docs/00-introduction.md)
1. [The CPU reference](docs/01-cpu-reference.md)
2. [Porting to the GPU](docs/02-gpu-port.md)
3. [Measuring performance correctly](docs/03-correct-measurement.md) — affinity, reproducibility, warm-up, comm-vs-compute
4. [Comparing communication variants](docs/04-communication-variants.md)
5. [Tuning the copy engines (SDMA vs blit)](docs/05-sdma-vs-blit.md)
6. [ROCm version & the SpMV API](docs/06-rocm-version-and-spmv.md)
7. [The optimized configuration](docs/07-optimization-path.md)
8. [Profiling communication & compute](docs/08-profiling.md) — rocprofv3, rocprof-compute, rocprofiler-systems, TAU, HPCToolkit, likwid, AMD uProf

In-depth measurement write-ups: [`CG-GPU/STUDY_REPORT.md`](CG-GPU/STUDY_REPORT.md) (OpenMPI, ROCm 6.4.1–7.13)
and [`CG-GPU/STUDY_REPORT_PrgEnv-amd.md`](CG-GPU/STUDY_REPORT_PrgEnv-amd.md) (HPE Cray EX, cray-mpich).
