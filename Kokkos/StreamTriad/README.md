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

## Building with Makefile

Every folder (Orig, Ver1, Ver2, Ver3, Ver4) also has a plain Makefile as an
alternative to CMake. `ROCM_PATH` and `Kokkos_DIR` default to the same paths
as the CMake flow, but are only used if not already set in the environment —
so load the module and export them first, same as for CMake:

```sh
module purge && module load rocm-new/10.1.0
export Kokkos_DIR=${HOME}/install/Kokkos_HIP
```

Then build and run:

```sh
cd Ver4
make          # build ./StreamTriad
make run      # run it (sets OMP_PROC_BIND=false and HSA_XNACK=1 for you)
make clean    # remove the binary and object files
```

## Hands-on exercise: build it yourself, step by step

This walks through turning a plain serial C++ loop into a Kokkos program
running on both CPU (OpenMP) and GPU (HIP). Each step's finished result
already exists in this directory (Ver1 through Ver4) as a reference solution
if you get stuck - try the modifications yourself on a copy of `Orig` before
peeking.

First, build Kokkos itself if you have not already - see the top-level
[Kokkos README](../README.md).

### Step 1: Test the serial baseline

```sh
cd Orig
cmake -B build ; cmake --build build ; ./build/StreamTriad
```

If the run fails (SEGV), try reducing the size of the arrays, by reducing
the value of the `nsize` variable in `StreamTriad.cc`.

### Step 2: Link against Kokkos

Add to `CMakeLists.txt`:

```
(add) find_package(Kokkos REQUIRED)
add_executable(StreamTriad ....)
(add) target_link_libraries(StreamTriad Kokkos::kokkos)
```

Retest with:

```sh
cmake -B build; cmake --build build
./build/StreamTriad
```

Check `Ver1` for a reference solution - these modifications have already
been made there (though note `Ver1`'s StreamTriad.cc is unmodified serial
code; linking against Kokkos here does not change behavior by itself).

### Step 3: Add Kokkos views for memory allocation of arrays

(peek at `Ver4/StreamTriad.cc` to see the end result)

Add the include:

```cpp
#include <Kokkos_Core.hpp>
```

Add initialize and finalize:

```cpp
Kokkos::initialize(argc, argv);  {

} Kokkos::finalize();
```

Replace the heap-allocated arrays with Kokkos views:

```cpp
int nsize=80000000;
Kokkos::View<double *> a( "a", nsize);
Kokkos::View<double *> b( "b", nsize);
Kokkos::View<double *> c( "c", nsize);
```

Rebuild and run:

```sh
cmake -B build ; cmake --build build ; ./build/StreamTriad
```

### Step 4: Add the Kokkos execution pattern - parallel_for

Change the `for` loops to Kokkos parallel fors.

At the start of the loop:

```cpp
Kokkos::parallel_for(nsize, KOKKOS_LAMBDA (int i) {
```

At the end of the loop, replace the closing brace with:

```cpp
});
```

Rebuild and run. Add environment variables as the Kokkos runtime message
suggests:

```sh
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8
```

How much speedup do you observe?

### Step 5: Add Kokkos timers

Add:

```cpp
Kokkos::Timer timer;
timer.reset(); // for timer start
time_sum += timer.seconds();
```

Remove:

```cpp
#include <timer.h>
struct timespec tstart;
cpu_timer_start(&tstart);
time_sum += cpu_timer_stop(tstart);
```

### Step 6: Measure OpenMP performance

Find out how many virtual cores are on your CPU:

```sh
lscpu
```

First run with a single thread (`OMP_NUM_THREADS=1`) and record the average
runtime, then run the OpenMP version with all available threads and record
the average runtime. How much speedup do you observe?

### Portability exercise: run on AMD GPUs

Rebuild StreamTriad against the HIP-enabled Kokkos install instead of the
OpenMP one (see the "Build and execute" section above for the full
module/env setup):

```sh
export Kokkos_DIR=${HOME}/install/Kokkos_HIP
cmake -B build -S . -DGPU_TARGETS=gfx942
cmake --build build
export OMP_PROC_BIND=false
export HSA_XNACK=1
./build/StreamTriad
```

Compare your result against the `Ver4` timings already measured above - how
does the GPU compare to the CPU/OpenMP version?
