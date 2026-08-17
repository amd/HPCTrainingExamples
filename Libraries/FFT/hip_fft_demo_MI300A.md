---
title: "Hands-On: FFT on MI300A — rocFFT / hipFFT + heFFTe"
subtitle: "MPO Training Workshop — Sept 8, 2026"
author: "MPO Training"
date: "September 8, 2026"
geometry: margin=1in
fontsize: 11pt
colorlinks: true
---

# FFT on MI300A — Hands-On Exercise

This exercise walks the AMD FFT stack from a single-device transform up to a
distributed multi-node 3D FFT:

1. **rocFFT** — 1D batched and 3D complex-to-complex, round-trip verified.
2. **hipFFT** — the same 1D transform through the portable (cuFFT-compatible) API.
3. **heFFTe** — distributed 3D FFT across ranks (rocFFT backend), launched with
   `mpirun`.
4. **APU unified memory** — initialize on the CPU, transform on the GPU, no copies.
5. Profiling (Appendix A), including the multi-node all-to-all cost.

> Build/PDF note: render with `pandoc hip_fft_demo_MI300A.md -o fft_demo.pdf`.

Environment (AAC / MI300A, `gfx942`):

```bash
module load rocm/7.2.4 openmpi     # rocFFT, hipFFT ship with ROCm; OpenMPI for MPI
export HSA_XNACK=1
```

---

## 0. FFT background you need for the exercise

- A 3D FFT is three sets of **batched 1D FFTs** (X, then Y, then Z) with a
  **transpose** between passes. On one device the transpose is a local reorder;
  **across nodes it becomes an all-to-all** — the dominant cost at scale.
- **Plan once, execute many.** Twiddle factors are precomputed at plan time.
- **The inverse is unnormalized**: divide by the transform length (N for 1D, the
  product of the dimensions for 3D).
- Favor sizes that factor into small primes (2,3,5,7); large primes fall back to
  Bluestein and run slower.

---

## 1. rocFFT — 1D batched C2C (single APU)

Complete program: `rocfft_c2c.hip`. Forward + inverse in-place on unified memory,
reporting the round-trip error and the work-buffer size.

Key API sequence:

```cpp
rocfft_setup();
rocfft_plan_create(&fwd, rocfft_placement_inplace,
    rocfft_transform_type_complex_forward, rocfft_precision_double,
    1 /*dim*/, &len, batch, nullptr);
rocfft_execution_info_create(&info);
rocfft_plan_get_work_buffer_size(fwd, &wb);          // query, allocate, attach
rocfft_execution_info_set_work_buffer(info, wbuf, wb);
void* ibuf[1] = { data };                            // unified-memory pointer
rocfft_execute(fwd, ibuf, nullptr, info);            // out=NULL => in-place
```

Get an allocation on a compute node

```bash
salloc -N 1 --gpus=1 --time=00:30:00 -p PPAC_MI300A_SPX
```

Set up the environment for the example

```bash
module load rocm
export HSA_XNACK=1
```

Build & run:

```bash
make rocfft_c2c
./rocfft_c2c 1048576 1
./rocfft_c2c 65536 64          # 64 batched transforms of length 65536
```

**Validated (MI300A / ROCm 7.2.4):**

```text
rocFFT C2C  N=1048576 batch=1  round-trip max_err=1.887e-15  work_buf=16777216 B
rocFFT C2C  N=65536 batch=64   round-trip max_err=1.460e-15  work_buf=67108864 B
```

### Exercises
- **E1.1** Confirm the round-trip error stays ~1e-15 as you vary N. Try a large
  **prime** N (e.g. 1000003) and watch the work-buffer/time jump (Bluestein path).
- **E1.2** Sweep `batch` at fixed N; plot transforms/sec. Where does batching stop
  helping?
- **E1.3 (APU)** Verify zero `hipMemcpy` in a trace (Appendix A.1); compare against
  an explicit `hipMalloc`+`hipMemcpy` variant.

---

## 2. rocFFT — 3D C2C (single APU)

Complete program: `rocfft_3d.hip`. An `N×N×N` cube, forward + inverse, round-trip
verified. Note the `length[3]` array is **fastest dimension first**, and the
work-buffer grows quickly with N.

```bash
make rocfft_3d
./rocfft_3d 128
./rocfft_3d 256
```

**Validated:**

```text
rocFFT 3D  128x128x128  round-trip max_err=1.776e-15  work_buf=33684480 B
rocFFT 3D  256x256x256  round-trip max_err=1.816e-15  work_buf=268696576 B
```

### Exercises
- **E2.1** Find the largest cube that fits in one APU's HBM (watch `work_buf` plus
  2× the data for out-of-place). This limit is exactly what motivates multi-node.
- **E2.2** Switch to an R2C/C2R transform (`rocfft_transform_type_real_forward`)
  and account for the ~2× memory savings from Hermitian symmetry.

---

## 3. hipFFT — the portable API

Complete program: `hipfft_c2c.hip`. Same 1D transform via the cuFFT-compatible
interface — identical source builds on AMD (rocFFT) and NVIDIA (cuFFT).

```cpp
hipfftHandle plan;
hipfftPlan1d(&plan, N, HIPFFT_Z2Z, batch);
hipfftExecZ2Z(plan, data, data, HIPFFT_FORWARD);
hipfftExecZ2Z(plan, data, data, HIPFFT_BACKWARD);
```

```bash
hipcc -O3 --offload-arch=gfx942 hipfft_c2c.hip -lhipfft -o hipfft_c2c
./hipfft_c2c 1048576 1
```

**Validated:** `hipFFT Z2Z  N=1048576 batch=1  round-trip max_err=1.887e-15`
(identical to rocFFT — hipFFT dispatches to rocFFT underneath).

### Exercises
- **E3.1** Diff `hipfft_c2c.hip` against `rocfft_c2c.hip`: what did the portability
  layer hide (work buffer, setup/cleanup, execution info)?
- **E3.2** Port a small cuFFT snippet by mechanical `cufft`→`hipfft` renaming and
  build it here.

---

## 4. heFFTe — distributed 3D FFT (multi-node, `mpirun`)

Reference program: `heffte_3d.cpp`. heFFTe splits the global cube across ranks,
runs local rocFFT transforms, and does the inter-rank transposes as MPI all-to-all.
You provide each rank's local input/output box; heFFTe builds the rest.

Core pattern (validated against heFFTe 2.4.1, ROCm backend):

```cpp
using backend_tag = heffte::backend::rocfft;
using cpx = std::complex<double>;

heffte::box3d<> world({0,0,0}, {N-1,N-1,N-1});
auto grid      = heffte::proc_setup_min_surface(world, nranks);   // balanced grid
auto in_boxes  = heffte::split_world(world, grid);
auto out_boxes = heffte::split_world(world, grid);
heffte::fft3d<backend_tag> fft(in_boxes[rank], out_boxes[rank], MPI_COMM_WORLD);

// Host -> device (malloc + memcpy) and back use the transfer helpers:
std::vector<cpx> host_in(fft.size_inbox());              /* fill on CPU */
heffte::gpu::vector<cpx> gpu_in = heffte::gpu::transfer().load(host_in);
heffte::gpu::vector<cpx> gpu_out(fft.size_outbox());
heffte::fft3d<backend_tag>::buffer_container<cpx> work(fft.size_workspace());

fft.forward (gpu_in.data(),  gpu_out.data(), work.data(), heffte::scale::none);
fft.backward(gpu_out.data(), gpu_in.data(),  work.data(), heffte::scale::full);
std::vector<cpx> host_back = heffte::gpu::transfer::unload(gpu_in);
```

> **Two gotchas we hit and fixed** (both in `heffte_3d.cpp`):
> 1. Destroy every heFFTe object *before* `MPI_Finalize()` — `fft3d`'s destructor
>    frees a duplicated communicator, so wrap the work in a `{ ... }` scope.
> 2. heFFTe GPU calls are asynchronous. Call `hipDeviceSynchronize()` before you
>    stop the timer, otherwise a single-rank run (no all-to-all to force a sync)
>    reports ~0 s.

### 4.1 Build heFFTe with the ROCm backend (one time, login node)

On the AAC6 system, you may find a module that already has a module built. Check for
`module avail` and look for heffte/2.4.1 module.

If there is not a pre-built module, to build heFFTe against rocFFT:

```bash
module load rocm/7.2.4 openmpi
git clone --depth 1 https://github.com/icl-utk-edu/heffte.git
cd heffte && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=hipcc \
      -DHeffte_ENABLE_ROCM=ON -DHeffte_ENABLE_FFTW=OFF \
      -DAMDGPU_TARGETS=gfx942 -DCMAKE_HIP_ARCHITECTURES=gfx942 \
      -DCMAKE_INSTALL_PREFIX=$HOME/heffte-rocm ..
make -j16 install
export HEFFTE_ROOT=$HOME/heffte-rocm
```

This yields **heFFTe 2.4.1** with `Heffte_ENABLE_ROCM=ON` and (by default here)
`Heffte_ENABLE_GPU_AWARE_MPI=ON`. The library installs to
`$HOME/heffte-rocm/{include,lib}` (shared `libheffte.so`), and CMake auto-detects
the ROCm-aware OpenMPI 5.0.10 `mpicxx`. Configure+build+install took ~1 min total.

### 4.2 Build and launch the exercise in an interactive Slurm allocation

Get an interactive allocation

```bash
salloc -N 1 --ntasks=4 --cpus-per-task=1 --gpus=4 -t 01:00:00 [-p <slurm queue>]
```

Build the `heffte_3d` example

```bash
make heffte_3d
```

Add the heFFTe library to the `LD_LIBRARY_PATH`

```bash
export LD_LIBRARY_PATH=$HOME/heffte-rocm/lib:$LD_LIBRARY_PATH
```

Run the example with the mpirun command on 4 MPI ranks

```bash
mpirun -np 4 --map-by ppr:4:node ./heffte_3d 512
```

exit the allocation

### 4.3 Running in an Slurm sbatch script

Launch with **`mpirun`** via a batch script (adapt `run_multinode.sbatch`):

The `libheffte.so` path is not baked into the binary's RPATH, so export it at
run time (`run_heffte.sbatch` does this):

```bash
#!/bin/bash
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:15:00
module load rocm/7.2.4 openmpi
export HSA_XNACK=1
export LD_LIBRARY_PATH=$HOME/heffte-rocm/lib:$LD_LIBRARY_PATH
mpirun -np "${SLURM_NTASKS}" --map-by ppr:4:node \
       -x HSA_XNACK -x LD_LIBRARY_PATH ./heffte_3d 512
```

Validated on MI300A (`gfx942`), 4 ranks / 1 node:

```
heFFTe 3D  512x512x512  ranks=4  grid=1x2x2  fwd+bwd=0.2172 s  round-trip max_err=2.113e-15
```

Round-trip error stays at the ~1e-15 double-precision floor across all rank
counts — the distributed transform is numerically correct.

### 4.4 Weak-scaling results (measured, ~256³ points per rank)

Fixed per-rank volume (~16.8 M complex-double points), growing the global cube
with the rank count. Steady-state average of 5 fwd+bwd round trips after a warm-up
call (`run_heffte_scaling.sbatch`):

| Ranks | Nodes | Global cube | Proc grid | fwd+bwd (avg) | round-trip err |
|------:|------:|------------:|:---------:|--------------:|---------------:|
| 1     | 1     | 256³        | 1×1×1     | ~0 s (local)  | 6.4e-15        |
| 2     | 1     | 324³        | 1×1×2     | 0.020 s       | 1.0e-14        |
| 4     | 1     | 400³        | 1×2×2     | 0.025 s       | 7.4e-15        |
| 8     | 2     | 512³        | 2×2×2     | **23.2 s**    | 9.9e-15        |

**The teaching point.** Intra-node scaling (1→4 APUs) is nearly flat: the
all-to-all rides the on-package fabric / xpmem shared memory, so doubling ranks
barely moves the wall-clock. Crossing to a **second node** collapses performance
by ~1000× — the inter-node all-to-all is the wall.

We confirmed this is the **fabric**, not the software path, by toggling heFFTe's
`plan_options.use_gpu_aware` (env `HEFFTE_GPU_AWARE`, see `run_heffte_comm.sbatch`):

| 8-rank / 2-node 512³ | fwd+bwd (avg of 5) |
|:---------------------|-------------------:|
| GPU-aware MPI ON     | 23.39 s            |
| host-staged (OFF)    | 23.95 s            |

Identical either way ⇒ this testbed's inter-node link (non-RDMA) is bandwidth-bound,
so the message never mattered. On a system with GPU-RDMA (InfiniBand/RoCE +
GPUDirect) the GPU-aware path would win and the 8-rank point would land far closer
to the intra-node trend. **This is exactly the kind of measurement to make before
committing an FFT-heavy code to multi-node runs.**

### 4.5 Exercises
- **E5.1** Reproduce the weak-scaling table; add a strong-scaling run (fix 512³
  global, grow ranks) and plot speedup.
- **E5.2 (communication)** Compare `proc_setup_min_surface` against a forced slab
  grid (`{nranks,1,1}`); measure the all-to-all difference.
- **E5.3** Profile the 8-rank run and separate compute from communication (see
  Appendix A.5). Confirm the inter-node all-to-all dominates.
- **E5.4 (alt)** ROCm 7.2.4 ships **hipFFT-MP** (`<hipfft/hipfftMp.h>`). Sketch the
  same distributed 3D FFT with hipFFT-MP and compare the programming model to
  heFFTe (native/minimal-deps vs. turnkey solver).

---

## 5. What "good" looks like
- Single device: round-trip error ~1e-15 (double); plan reuse; no `hipMemcpy` on
  the APU path.
- 3D single device: work-buffer + data fit in HBM — the ceiling that motivates
  multi-node.
- heFFTe: correct round-trip across ranks (~1e-15); intra-node weak-scaling
  nearly flat, with the inter-node all-to-all as the dominant cost at scale.

---

# Appendix A — On-Your-Own Profiling

## A.1 Confirm the copy-free APU path (rocprofv3)
```bash
rocprofv3 --sys-trace -- ./rocfft_c2c 1048576 1
# Look for MEMCPY rows (should be absent on the unified-memory path) and the
# rocFFT kernels (Stockham/transpose stages).
```

## A.2 Per-kernel timing (rocprofv3)
```bash
rocprofv3 --kernel-trace --stats -- ./rocfft_3d 256
# Which stages dominate: the 1D FFT passes or the internal transposes?
```

## A.3 Roofline (rocprof-compute)
```bash
rocprof-compute profile -n fft3d -- ./rocfft_3d 256
rocprof-compute analyze -p workloads/fft3d/MI300A/
# Confirm the FFT kernels are HBM-bandwidth bound; note achieved BW vs peak.
```

## A.4 Power / energy (rocprof-sys)
```bash
rocprof-sys-run -- ./rocfft_3d 256    # with ROCPROFSYS_USE_ROCM_SMI=ON
# Integrate power over the run for energy-per-transform.
```

## A.5 Multi-node all-to-all fraction (heFFTe)
- `heffte_3d.cpp` already reports a warm, averaged fwd+bwd time via `MPI_Wtime`
  (with a `hipDeviceSynchronize()` so async GPU work is captured).
- On our testbed the 1→4 rank (single-node) times were ~0.02–0.03 s, but the
  8-rank (2-node) time jumped to ~23 s — the inter-node all-to-all dominated.
- Toggling `HEFFTE_GPU_AWARE=0/1` (`run_heffte_comm.sbatch`) gave the same 23 s,
  proving the inter-node link (non-RDMA), not the MPI path, is the bottleneck.
- Add per-node power sampling before `mpirun` (single-task-per-node `srun` only to
  fan out `amd-smi monitor --power`, not to launch the job).
- Report: fraction of time in communication vs. local FFT as ranks grow.

---

## Appendix B — File manifest

| File | Purpose |
|---|---|
| `rocfft_c2c.hip` | 1D batched C2C, forward+inverse round-trip (validated ~1e-15) |
| `rocfft_3d.hip`  | 3D C2C round-trip (validated ~1e-15) |
| `hipfft_c2c.hip` | 1D C2C via portable hipFFT API (validated ~1e-15) |
| `heffte_3d.cpp`  | Distributed 3D FFT, rocFFT backend (validated on 1/2/4/8 ranks) |
| `run_heffte.sbatch` | Single distributed run (`sbatch run_heffte.sbatch [N]`) |
| `run_heffte_scaling.sbatch` | Weak-scaling sweep 1→2→4→8 ranks |
| `run_heffte_comm.sbatch` | GPU-aware vs host-staged all-to-all comparison |

## References
- rocFFT / hipFFT documentation (ROCm 7.2.4).
- heFFTe: Ayala et al., *Heffte: Highly Efficient FFT for Exascale* (ICL/UTK).
