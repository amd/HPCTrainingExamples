# kokkos_check_apu_enabled

Standalone sanity check that a Kokkos install is built with the HIP backend
and the expected AMD GPU architecture — compile-time asserts plus a runtime
dump of `Kokkos::print_configuration()`. Useful for quickly confirming a
Kokkos install actually matches the node you're on (MI300A APU vs MI300X
discrete, both `gfx942`) before debugging a larger build.

## Source files

- `kokkos_check_apu_enabled.cpp` — asserts `KOKKOS_ARCH_AMD_GFX942_APU` (or
  `KOKKOS_ARCH_AMD_GFX942` for discrete GPUs), `KOKKOS_ENABLE_HIP`, and
  `KOKKOS_ARCH_AMD_GPU` are defined at compile time, then prints the full
  Kokkos runtime configuration and a PASS/FAIL summary.
- `CMakeLists.txt` — locates Kokkos via `find_package(Kokkos REQUIRED)`, uses
  `kokkos_check()` to verify the `AMD_GFX942_APU` arch and `HIP` backend are
  enabled, and places the built binary in the repo root
  (`RUNTIME_OUTPUT_DIRECTORY`).
- `Makefile` — builds the same check directly with `amdclang++`, without
  going through CMake's `kokkos_check()`; defines `KOKKOS_CHECK_EXPECT_APU=1`
  to match this cluster's APU-built Kokkos install.

## Building and running

Via CMake:

```sh
module purge && module load rocm-new/10.1.0
export Kokkos_DIR=${HOME}/install/Kokkos_HIP
export CC=${ROCM_PATH}/llvm/bin/amdclang
export CXX=${ROCM_PATH}/llvm/bin/amdclang++

cmake -B build -S . -DGPU_TARGETS=gfx942
cmake --build build

export OMP_PROC_BIND=false
export HSA_XNACK=1
./kokkos_check_apu_enabled
```

Via Makefile (equivalent, no CMake involved):

```sh
module purge && module load rocm-new/10.1.0

make
export OMP_PROC_BIND=false
export HSA_XNACK=1
make run
```

Both require a GPU-allocated node to run successfully, e.g.:

```sh
salloc --partition=192C4G1H_MI300A_RHEL9_A1 -N1 --gpus=1 --ntasks=8 -t 03:00:00
```
