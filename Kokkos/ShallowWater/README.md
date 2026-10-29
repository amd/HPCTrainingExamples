# ShallowWater (Kokkos)

2D shallow water equation solver (Bob Robey, LANL), progressively ported from
serial C++ to Kokkos.

## Source files

- `ShallowWater.cc` — original serial baseline (no Kokkos).
- `ShallowWater_par1.cc` — ports the data arrays to `Kokkos::View`, logic still serial.
- `ShallowWater_par2.cc` — adds `Kokkos::parallel_for` over `RangePolicy`/`MDRangePolicy`.
- `ShallowWater_par3.cc` — adds `Kokkos::parallel_reduce` for reduction steps (e.g. mass/timestep).
- `ShallowWater_par4.cc` — same as par3, but swaps timestep buffers by copying View contents each iteration.
- `ShallowWater_par5.cc` — like par4, but swaps View handles (pointer swap) instead of copying data.
- `memory.c` / `memory.h` — 1D/2D heap allocation helpers (`dvector`, `malloc2D`) used by the serial baseline.
- `timer.c` / `timer.h` — `clock_gettime`-based wall-timer used by `ShallowWater.cc` and `par1`-`par3`.
- `graphics/` — optional FreeGLUT-based visualization, built only when `ENABLE_GRAPHICS` is set.
- `kokkos_cmake_options.txt` — reference list of Kokkos CMake options (CUDA/HIP/arch flags); not consumed by the build.
- `CMakeLists.txt` — defines all six executables and locates Kokkos via `find_package(Kokkos REQUIRED)`.

`ShallowWater_par1`-`par5` pick their Kokkos execution space via an `ExecSpace`
alias near the top of each file: `Kokkos::DefaultExecutionSpace` by default
(HIP, since that's enabled in the linked Kokkos install), or `Kokkos::Serial`/
`Kokkos::OpenMP` if compiled with `-DFORCE_SERIAL`/`-DFORCE_OPENMP`.

## Build scripts

- `build_serial_version.sh` — configures/builds in `serial_build/`, compiling with `-DFORCE_SERIAL`.
- `build_openmp_version.sh` — configures/builds in `openmp_build/`, compiling with `-DFORCE_OPENMP`.
- `build_hip_version.sh` — configures/builds in `hip_build/` for AMD GPUs (`gfx942`); uses the HIP (default) execution space.
- `build_cuda_version.sh` — configures/builds in `cuda_build/` for NVIDIA GPUs (Volta); unused on this AMD cluster.

Each script creates its build directory, runs `cmake`/`make -j 8`, then runs every built executable in turn.
All three AMD scripts `module purge` then `module load rocm-new/10.1.0`, and link against the
single pre-built Kokkos install at `$HOME/install/Kokkos_HIP` (built with Serial, OpenMP, and HIP all enabled).

**Caveat:** `Kokkos::initialize()` always initializes *every* backend compiled
into the linked Kokkos library, not just the one your code uses. Since HIP is
compiled into `Kokkos_HIP`, the `ShallowWater_par*` binaries always probe for a
ROCm-capable GPU at startup and abort with `hipErrorNoDevice` if none is
present — even when built with `-DFORCE_SERIAL`/`-DFORCE_OPENMP`. So
`build_serial_version.sh`/`build_openmp_version.sh` only run the Kokkos
binaries successfully on a node with a GPU allocated; the plain `ShallowWater`
binary (no Kokkos) always runs fine anywhere.

## Running the builds

```sh
# Serial execution space (needs a GPU-allocated node to run ShallowWater_par*, see caveat above)
./build_serial_version.sh

# OpenMP execution space (same caveat)
./build_openmp_version.sh

# HIP backend (AMD GPU, e.g. MI300A/gfx942) - needs a GPU allocated, e.g. via salloc/srun
./build_hip_version.sh
```

## Running on an allocated compute node

The `ShallowWater_par*` binaries require a GPU, so request an interactive node
first, then build/run from inside that allocation:

```sh
salloc --partition=192C4G1H_MI300A_RHEL9_A1_CPX -n1 --gpus=1 --ntasks=8 -t 03:00:00

# once dropped onto the allocated node:
./build_hip_version.sh

# or just run an already-built binary directly:
./hip_build/ShallowWater_par5
```
