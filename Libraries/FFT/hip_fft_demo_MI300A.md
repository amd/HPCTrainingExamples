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
3. **hipFFTW** — the same 1D transform through the portable (FFTW3-compatible) API;
   a legacy CPU FFTW3 program runs on the GPU with no source changes.
4. **hipFFT-MP** — native multi-process distributed 3D FFT over MPI, built into
   ROCm 7.2.4 (`<hipfft/hipfftMp.h>`); the cuFFTMp-equivalent, no external solver.
5. **heFFTe** — distributed 3D FFT across ranks (rocFFT backend), launched with
   `mpirun`.
6. **APU unified memory** — initialize on the CPU, transform on the GPU, no copies.
7. Profiling (Appendix A), including the multi-node all-to-all cost.

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

## 4. hipFFTW — the FFTW3 drop-in (single APU)

Complete program: `fftw_c2c.c`. The same 1D transform as §1–§3, but written in the
**plain FFTW3 API** — no HIP, no device pointers, no rocFFT/hipFFT calls. hipFFTW
(shipped inside hipFFT starting in ROCm 7.1.0, production in 7.2.x) exports the
FFTW3 symbols (`fftw_*` / `fftwf_*`), so a legacy CPU FFTW3 code runs on an AMD GPU
with **zero source changes**. The only difference from a CPU build is one include
and one link flag:

```text
CPU (FFTW3):     #include <fftw3.h>            ...  -lfftw3
GPU (hipFFTW):   #include <hipfft/hipfftw.h>   ...  -lhipfftw
```

Unlike hipFFT, hipFFTW does **not** require its buffers to be GPU-visible — it
stages `host<->device` under the hood. On an APU (unified HBM) that staging is cheap,
so the relink alone buys GPU acceleration. This is the easiest possible on-ramp for
a CPU-only FFTW3 code base.

```c
#include <hipfft/hipfftw.h>
fftw_complex* data = fftw_alloc_complex(N);
fftw_plan fwd = fftw_plan_dft_1d(N, data, data, FFTW_FORWARD,  FFTW_ESTIMATE);
fftw_plan bwd = fftw_plan_dft_1d(N, data, data, FFTW_BACKWARD, FFTW_ESTIMATE);
fftw_execute(fwd);            // or fftw_execute_dft(fwd, in, out) to reuse the plan
fftw_execute(bwd);            // inverse is unnormalized: divide by N
```

Build & run. Note the `-x c++` on the compiler line: the ROCm 7.2.4 
hipFFTW header includes C++ standard
headers (`<cstddef>`, `<cstdlib>`), so the translation unit including it must be
compiled as C++ (or named `.cpp`). The FFTW3 source itself is unchanged.

```bash
salloc -N 1 --gpus=1 --time=00:30:00 -p PPAC_MI300A_SPX
```

Set up the environment for the example. 

```bash
module load rocm
export HSA_XNACK=1
```

```bash
make fftw_c2c
# equivalently:
#   hipcc -O3 --offload-arch=gfx942 -x c++ -DUSE_HIPFFTW fftw_c2c.c -lhipfftw -o fftw_c2c
./fftw_c2c 1048576 1
./fftw_c2c 65536 64          # 64 batched transforms of length 65536
```

### 4a. CPU baseline — the *same* source on reference FFTW3

The whole point of the FFTW3 API is portability, so `fftw_c2c.c` also builds as an
ordinary CPU program against reference FFTW3 — **no HIP, no GPU**. The single source
selects its backend with the `USE_HIPFFTW` macro (set by the Makefile): with it, the
hipFFTW header + `-lhipfftw`; without it, `<fftw3.h>` + `-lfftw3`. Nothing else in
the source changes. CPU FFTW3 ships as the `fftw/3.3.10` module, which becomes
available once the `rocm` module is loaded:

```bash
module load rocm       # exposes the fftw module (and heffte)
module load fftw       # fftw/3.3.10 — sets FFTW_ROOT / FFTW_PATH, include & lib paths
export HSA_XNACK=1
```

```bash
make fftw_c2c_cpu
# equivalently:
#   gcc -O3 -I$FFTW_ROOT/include fftw_c2c.c -L$FFTW_ROOT/lib -lfftw3 -lm -o fftw_c2c_cpu
./fftw_c2c_cpu 1048576 1
./fftw_c2c_cpu 65536 64          # 64 batched transforms of length 65536
./fftw_c2c_cpu 1000003 1         # prime N
```

**Validated (MI300A host CPU, FFTW 3.3.10):** round-trip at the double-precision
floor, matching the GPU hipFFTW run bit-for-bit in magnitude:

```text
FFTW C2C  N=1048576 batch=1   round-trip max_err=1.333e-15
FFTW C2C  N=65536 batch=64    round-trip max_err=8.960e-16
FFTW C2C  N=1000003 batch=1   round-trip max_err=2.740e-15
```

**Validated (MI300A / ROCm 7.2.4, hipFFT 1.0.22):** round-trip `max_err` at the
~1e-15 double floor, *identical* to `rocfft_c2c` / `hipfft_c2c` on the same sizes —
hipFFTW dispatches to rocFFT underneath:

```text
hipFFTW C2C  N=1048576 batch=1   round-trip max_err=1.887e-15
hipFFTW C2C  N=65536 batch=64    round-trip max_err=1.460e-15
hipFFTW C2C  N=1000003 batch=1   round-trip max_err=6.062e-15   # prime -> Bluestein
```

> - The header must be compiled as C++ (see the `-x c++` note above) — a genuinely
>   C-only build (`gcc -lfftw3` -> `-lhipfftw`) does *not* compile as-is today.
> - Only the **basic** plans ship in 7.2.4 (`fftw_plan_dft_1d/2d/3d`,
>   `fftw_plan_dft`, and the r2c/c2r variants) plus `fftw_execute[_dft]`. The
>   **advanced** (`fftw_plan_many_dft`) and **guru** interfaces are not present yet,
>   so batching is done by reusing a basic plan with `fftw_execute_dft`.
> - Plan flags (`FFTW_MEASURE`, `FFTW_PATIENT`, wisdom, `FFTW_CONSERVE_MEMORY`) are
>   currently **ignored** — configurations are chosen by heuristic, no measurement
>   phase — and input preservation is not guaranteed. Interleaved complex only.

### Exercises
- **E4.1** Diff `fftw_c2c.c` against `hipfft_c2c.hip`: the FFTW source has no HIP or
  device code at all — hipFFTW hid the GPU entirely. What did you give up vs. the
  explicit hipFFT path (control over device buffers, streams, work areas)?
- **E4.2 (portability)** Build the CPU baseline (`make fftw_c2c_cpu`, §4a) from the
  *identical* source against reference FFTW3 (`-lfftw3`, `#include <fftw3.h>`) and
  compare results and wall-clock against the GPU hipFFTW run on the APU. Only the
  Makefile/`USE_HIPFFTW` flag changes. (Uses the `fftw/3.3.10` module.)
- **E4.3** Add an R2C/C2R variant (`fftw_plan_dft_r2c_1d` / `fftw_plan_dft_c2r_1d`)
  and account for the Hermitian-symmetry (~2×) memory savings.
- **E4.4** Confirm `FFTW_MEASURE` is currently a no-op in hipFFTW (time plan
  creation with `FFTW_ESTIMATE` vs. `FFTW_MEASURE`). What does that imply for
  porting a code that relies on FFTW wisdom/measurement?

---

## 5. hipFFT-MP — native distributed 3D FFT (multi-process, MPI)

Complete program: `hipfftmp_3d.cpp`. hipFFT-MP is the **multi-process** distributed
FFT that ships *inside* hipFFT starting in ROCm 7.2.4 (`<hipfft/hipfftMp.h>`) — the
AMD analog of NVIDIA's cuFFTMp. Where heFFTe (§6) is a separate solver you build
and link, hipFFT-MP needs only the ROCm stack: you attach an MPI communicator to an
ordinary hipFFT plan and the library performs the pencil/slab decomposition and the
inter-rank all-to-all for you.

The programming model is the single-process multi-GPU `hipfftXt` flow with one extra
call — `hipfftMpAttachComm` — before the plan is made:

```cpp
hipfftHandle plan;
hipfftCreate(&plan);
MPI_Comm comm = MPI_COMM_WORLD;
hipfftMpAttachComm(plan, HIPFFT_COMM_MPI, &comm);        // <-- the MP step
size_t work;
hipfftMakePlan3d(plan, N, N, N, HIPFFT_Z2Z, &work);     // built-in decomposition

hipLibXtDesc* desc;                                      // distributed device memory
hipfftXtMalloc(plan, &desc, HIPFFT_XT_FORMAT_INPLACE);
hipfftXtMemcpy(plan, desc, host_slab, HIPFFT_COPY_HOST_TO_DEVICE);   // scatter

hipfftXtExecDescriptor(plan, desc, desc, HIPFFT_FORWARD);
hipfftXtExecDescriptor(plan, desc, desc, HIPFFT_BACKWARD);

hipfftXtMemcpy(plan, host_slab, desc, HIPFFT_COPY_DEVICE_TO_HOST);   // gather
hipfftXtFree(desc);
hipfftDestroy(plan);
```

> **Notes / gotchas:**
> 1. Attach the communicator **before** `hipfftMakePlan3d` — the plan is what gets
>    built distributed. (The MP API is marked *Experimental* in the headers.)
> 2. With the **built-in** decomposition each rank owns a contiguous slab of the
>    slowest dimension, so this example requires `N % nranks == 0`. `hipfftXtMemcpy`
>    scatters/gathers between the per-rank host slab and the distributed descriptor.
> 3. The inverse is **unnormalized**: divide the round-trip result by `N³`.
> 4. For a hand-tuned layout use `hipfftXtSetDistribution` (custom decomposition):
>    each rank declares its input/output *brick* by lower/upper global coordinates.

### 5.1 Build and launch in an interactive Slurm allocation

Get an interactive allocation

```bash
salloc -N 1 --ntasks=4 --cpus-per-task=1 --gpus=4 -t 01:00:00 [-p <slurm queue>]
```

Set up the environment and build (hipFFT-MP ships with ROCm — no external library)

```bash
module load rocm/7.2.4 openmpi
export HSA_XNACK=1
make hipfftmp_3d
```

Run on 4 MPI ranks (one GPU per rank)

```bash
mpirun -np 4 --map-by ppr:4:node ./hipfftmp_3d 512
```

Or submit the batch script (`run_hipfftmp.sbatch`):

```bash
sbatch run_hipfftmp.sbatch 512
```

Expected output (round-trip error at the double-precision floor):

```text
hipFFT-MP 3D  512x512x512  ranks=4  slab=128x512x512  fwd+bwd=... s  round-trip max_err=~1e-15
```

### 5.2 Exercises
- **E5.1** Diff the plan setup against `heffte_3d.cpp`: hipFFT-MP adds one call
  (`hipfftMpAttachComm`) to the familiar `hipfftXt` flow, whereas heFFTe asks you to
  build the process grid and boxes explicitly. Which model do you prefer, and why?
- **E5.2** Sweep ranks 1→2→4 at a fixed 512³ cube (strong scaling) and compare the
  fwd+bwd time to the heFFTe numbers in §6.4 on the same node.
- **E5.3 (portability)** hipFFT-MP mirrors cuFFTMp call-for-call
  (`cufftMpAttachComm` → `hipfftMpAttachComm`, etc.). Take a small cuFFTMp snippet
  and port it by mechanical `cufft`→`hipfft` renaming.
- **E5.4 (custom layout)** Replace the built-in decomposition with
  `hipfftXtSetDistribution`, handing each rank an explicit input/output brick, and
  confirm the round-trip error is unchanged.

---

## 6. heFFTe — distributed 3D FFT (multi-node, `mpirun`)

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

> **Two cautions** (both in `heffte_3d.cpp`):
> 1. Destroy every heFFTe object *before* `MPI_Finalize()` — `fft3d`'s destructor
>    frees a duplicated communicator, so wrap the work in a `{ ... }` scope.
> 2. heFFTe GPU calls are asynchronous. Call `hipDeviceSynchronize()` before you
>    stop the timer, otherwise a single-rank run (no all-to-all to force a sync)
>    reports ~0 s.

### 6.1 Build heFFTe with the ROCm backend (one time, login node)

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

### 6.2 Build and launch the exercise in an interactive Slurm allocation

Get an interactive allocation

```bash
salloc -N 1 --ntasks=4 --cpus-per-task=1 --gpus=4 -t 01:00:00 [-p <slurm queue>]
```

Set up the environment

```bash
module load rocm heffte
export HSA_XNACK=1
```

Build the `heffte_3d` example

```bash
make heffte_3d
```

Build heFFTe library and add the heFFTe library to the `LD_LIBRARY_PATH`. Done by the heffte module

```bash
export LD_LIBRARY_PATH=$HOME/heffte-rocm/lib:$LD_LIBRARY_PATH
```

Run the example with the mpirun command on 4 MPI ranks

```bash
mpirun -np 4 --map-by ppr:4:node ./heffte_3d 512
```

Validated on MI300A (`gfx942`), 4 ranks / 1 node:

```
heFFTe 3D  512x512x512  ranks=4  grid=1x2x2  fwd+bwd=0.2172 s  round-trip max_err=2.113e-15
```

exit the allocation

### 6.3 Running in an Slurm sbatch script

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

Submit batch job

```
sbatch run_heffte.sbatch
```

Round-trip error stays at the ~1e-15 double-precision floor across all rank
counts — the distributed transform is numerically correct.

### 6.4 Weak-scaling results (measured, ~256³ points per rank)

Fixed per-rank volume (~16.8 M complex-double points), growing the global cube
with the rank count. Steady-state average of 5 fwd+bwd round trips after a warm-up
call (`run_heffte_scaling.sbatch`):

| Ranks | Nodes | Global cube | Proc grid | fwd+bwd (avg) | round-trip err |
|------:|------:|------------:|:---------:|--------------:|---------------:|
| 1     | 1     | 256³        | 1×1×1     | ~0 s (local)  | 6.4e-15        |
| 2     | 1     | 324³        | 1×1×2     | 0.020 s       | 1.0e-14        |
| 4     | 1     | 400³        | 1×2×2     | 0.025 s       | 7.4e-15        |
| 8     | 2     | 512³        | 2×2×2     | **23.2 s**    | 9.9e-15        |

**Note** Intra-node scaling (1→4 APUs) is nearly flat: the
all-to-all rides the on-package fabric / xpmem shared memory, so doubling ranks
barely moves the wall-clock. Crossing to a **second node** collapses performance
by ~1000× due to the inter-node all-to-all on the weak NICs on this system.

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

### 6.5 Exercises
- **E6.1** Reproduce the weak-scaling table; add a strong-scaling run (fix 512³
  global, grow ranks) and plot speedup.
- **E6.2 (communication)** Compare `proc_setup_min_surface` against a forced slab
  grid (`{nranks,1,1}`); measure the all-to-all difference.
- **E6.3** Profile the 8-rank run and separate compute from communication (see
  Appendix A.5). Confirm the inter-node all-to-all dominates.
- **E6.4 (alt)** Run the **hipFFT-MP** example (§5) on the same allocation and
  compare its programming model and fwd+bwd time to heFFTe here — native/minimal-deps
  (`hipfftMpAttachComm` on a stock hipFFT plan) vs. the turnkey solver.

---

## 7. What "good" looks like
- Single device: round-trip error ~1e-15 (double); plan reuse; no `hipMemcpy` on
  the APU path.
- 3D single device: work-buffer + data fit in HBM — the ceiling that motivates
  multi-node.
- hipFFT-MP: native distributed round-trip correct across ranks (~1e-15) with only
  the ROCm stack — one extra call (`hipfftMpAttachComm`) over the single-node path.
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
| `fftw_c2c.c` | 1D C2C in the plain FFTW3 API; `make fftw_c2c` runs it on the GPU via hipFFTW, `make fftw_c2c_cpu` builds the same source on CPU FFTW3 (`fftw` module) |
| `hipfftmp_3d.cpp` | Native distributed 3D FFT via hipFFT-MP (ROCm 7.2.4+, MPI) |
| `run_hipfftmp.sbatch` | Single hipFFT-MP run (`sbatch run_hipfftmp.sbatch [N]`) |
| `heffte_3d.cpp`  | Distributed 3D FFT, rocFFT backend (validated on 1/2/4/8 ranks) |
| `run_heffte.sbatch` | Single distributed run (`sbatch run_heffte.sbatch [N]`) |
| `run_heffte_scaling.sbatch` | Weak-scaling sweep 1→2→4→8 ranks |
| `run_heffte_comm.sbatch` | GPU-aware vs host-staged all-to-all comparison |

## References
- rocFFT / hipFFT / hipFFTW documentation (ROCm 7.2.4), including the hipFFT-MP
  multi-process API (`<hipfft/hipfftMp.h>`, `hipfftMpAttachComm`) and the FFTW3
  drop-in interface (`<hipfft/hipfftw.h>`).
- FFTW: Frigo & Johnson, *The Design and Implementation of FFTW3* (fftw.org).
- heFFTe: Ayala et al., *Heffte: Highly Efficient FFT for Exascale* (ICL/UTK).
