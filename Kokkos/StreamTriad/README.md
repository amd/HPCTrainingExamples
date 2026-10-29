# StreamTriad

Five progressive versions of the STREAM Triad benchmark (`c[i] = a[i] + scalar*b[i]`), showing the migration path from plain C++ to Kokkos.

## Folders

- **Orig** — Plain C++, heap-allocated arrays (`malloc`), serial CPU loops, no Kokkos.
- **Ver1** — Same as Orig: plain C++, heap-allocated arrays, serial CPU loops, no Kokkos.
- **Ver2** — Introduces Kokkos only for data (`Kokkos::View`); the triad and init loops are still plain serial `for` loops.
- **Ver3** — Fully Kokkos: `Kokkos::View` for data and `Kokkos::parallel_for` for both init and triad loops.
- **Ver4** — Same as Ver3, plus timing via `Kokkos::Timer`/`Kokkos::fence` instead of the custom `timer.c` utility.

## Builid and execute

Before building, load the ROCm toolchain used to build Kokkos, point CMake at that Kokkos install, and pin the compiler (otherwise CMake falls back to the Cray compiler wrapper and `find_package(Threads)` fails):

```sh
module purge && module load rocm-new/10.1.0
export Kokkos_DIR=${HOME}/install/Kokkos_HIP
export CC=${ROCM_PATH}/llvm/bin/amdclang
export CXX=${ROCM_PATH}/llvm/bin/amdclang++
```

Every folder (Orig, Ver1, Ver2, Ver3, Ver4) builds the same way, via CMake, in two steps:

```sh
cd Ver1
cmake -B build -S . -DGPU_TARGETS=gfx942
cmake --build build
```

Before running the binary, set:

```sh
export OMP_PROC_BIND=false
export HSA_XNACK=1
```

Then run it:

```sh
./StreamTriad
```
