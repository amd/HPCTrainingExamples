# FFT on GPUs (HIP) — rocFFT / hipFFT / hipFFTW / hipFFT-MP / heFFTe

Hands-on FFT exercises for AMD GPUs (developed and tuned on the MI300A APU).
The exercise walks the AMD FFT stack from a single-device transform up to a
distributed multi-node 3D FFT.

The full step-by-step guide and results are in the companion write-up:

- [`hip_fft_demo_MI300A.md`](hip_fft_demo_MI300A.md) — the hands-on exercise guide
  (build, run, verify each transform; APU unified-memory notes; profiling appendix).

## What you will do

1. **rocFFT** — 1D batched and 3D complex-to-complex, round-trip verified.
2. **hipFFT** — the same 1D transform through the portable (cuFFT-compatible) API.
3. **hipFFTW** — the same 1D transform through the portable (FFTW3-compatible) API;
   a legacy CPU FFTW3 program runs on the GPU with no source changes.
4. **hipFFT-MP** — native multi-process distributed 3D FFT over MPI (built into
   ROCm 7.2.4, `<hipfft/hipfftMp.h>`); the cuFFTMp-equivalent, no external solver.
5. **heFFTe** — distributed 3D FFT across ranks (rocFFT backend), launched with `mpirun`.
6. **APU unified memory** — initialize on the CPU, transform on the GPU, no copies.
7. Profiling (Appendix A), including the multi-node all-to-all cost.

## Source files

- `rocfft_c2c.hip` — rocFFT 1D batched complex-to-complex
- `rocfft_3d.hip` — rocFFT 3D complex-to-complex
- `hipfft_c2c.hip` — hipFFT 1D batched (portable cuFFT-compatible API)
- `fftw_c2c.c` — one source, two builds: GPU via hipFFTW (`-DUSE_HIPFFTW`) and CPU via reference FFTW3
- `hipfftmp_3d.cpp` — native distributed 3D FFT (MPI-enabled hipFFT, ROCm 7.2.4+)
- `heffte_3d.cpp` — distributed 3D FFT using heFFTe (rocFFT backend)
- `heffte/` — the heFFTe library checkout (examples, tests, Python/Fortran bindings; see its own READMEs)
- `run_fft_tests.sh` — single-APU build + smoke tests
- `run_heffte.sbatch`, `run_heffte_scaling.sbatch`, `run_heffte_comm.sbatch`, `run_hipfftmp.sbatch` — Slurm launchers

## Build and run

```bash
module load rocm/7.2.4 openmpi     # rocFFT, hipFFT ship with ROCm; OpenMPI for MPI
export HSA_XNACK=1                  # APU: enable page migration for unified-memory paths

make                # build the single-device binaries (rocFFT / hipFFT / hipFFTW)
make hipfftmp_3d    # native distributed FFT (needs MPI)
make heffte_3d      # distributed FFT via heFFTe (set HEFFTE_ROOT; see demo sec 5.1)
make pdf            # render hip_fft_demo_MI300A.md to PDF (needs pandoc + LaTeX)
make clean

# quick smoke test on a single APU:
./run_fft_tests.sh
```

The Makefile auto-detects the GPU arch via `rocminfo` and builds with `hipcc`.
CPU FFTW3 baselines use the `fftw/3.3.10` module (`FFTW_PATH`/`FFTW_ROOT`).
heFFTe builds expect `HEFFTE_ROOT` (defaults to `$HOME/heffte-rocm`).

See [`hip_fft_demo_MI300A.md`](hip_fft_demo_MI300A.md) for the full exercise and
the profiling appendix.
