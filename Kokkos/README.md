# Kokkos

**Disclaimer**: these examples have been tested on AMD's internal cluster AAC7 (MI300A accelerators).

Instructions for building the [Kokkos](https://github.com/kokkos/kokkos) C++
performance-portability library from source, for use by the example projects
in this directory.

## Building Kokkos

```bash
git clone https://github.com/kokkos/kokkos Kokkos_build
cd Kokkos_build
```

Load a ROCm module if needed. Whichever module you use here must also be
loaded later when building the example projects against this Kokkos install
(e.g. the StreamTriad/ShallowWater examples in this repo were built and
tested with `rocm-new/10.1.0` - use the same one consistently for both steps).

```
module load rocm-new/10.1.0
```

Set CC and CXX in environment (bash shown)

```
export CC=amdclang
export CXX=amdclang++
```

Build Kokkos with OpenMP backend

```
cmake -B build_openmp -S . \
      -DCMAKE_INSTALL_PREFIX=${HOME}/install/Kokkos_OpenMP -DKokkos_ENABLE_SERIAL=On \
      -DKokkos_ENABLE_OPENMP=On ..

cmake --build build_openmp --target install --parallel 8

```
Build Kokkos with HIP backend (change CPU and GPU arch as necessary, see https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#cpu-architectures and https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#amd-gpus) *Note: OpenMP and HIP support can be configured in the same build if desired. Note2: modern CMake will possibly auto-detect a different GPU_TARGET, to avoid problems explicitly set the `-DGPU_TARGETS` to your architecture _without_ the `:sramecc+:xnack-` properties.*

```
cmake -B build_hip -S . \
      -DCMAKE_INSTALL_PREFIX=${HOME}/install/Kokkos_HIP -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_ZEN=ON -DKokkos_ARCH_AMD_GFX942_APU=ON \
      -DGPU_TARGETS="gfx942"

cmake --build build_hip --target install --parallel 8
```

Set Kokkos_DIR to point to the Kokkos package you want examples to use, e.g.:

```
export Kokkos_DIR=${HOME}/install/Kokkos_HIP
```

## Examples in this repository

- [StreamTriad/](StreamTriad/) — progressive CPU-to-GPU port of the STREAM
  Triad benchmark across five versions, with a step-by-step hands-on exercise.
- [ShallowWater/](ShallowWater/) — 2D shallow water equation solver,
  progressively ported from serial C++ to Kokkos.
- [kokkos_check_apu_enabled/](kokkos_check_apu_enabled/) — standalone sanity
  check confirming the architecture and backend configuration of a Kokkos install.
