
# Kokkos examples

## Stream Triad

### Step 1: Build a separate Kokkos package

**NOTE**: these exercises have been tested on MI210 and MI300A accelerators using a container environment.
To see details on the container environment (such as operating system and modules available) please see `README.md` on [this](https://github.com/amd/HPCTrainingDock) repo.

```bash
cd $HOME/HPCTraining/Examples
git clone https://github.com/kokkos/kokkos Kokkos_build
cd Kokkos_build
```

Load a ROCm module if needed

```
module load rocm/7.2.1
```

Set CC and CXX in environment (bash shown)

```
export CC=amdclang
export CXX=amdclang++
```

Build Kokkos with OpenMP backend

```
cmake -B build_openmp -S . \
      -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos_OpenMP -DKokkos_ENABLE_SERIAL=On \
      -DKokkos_ENABLE_OPENMP=On ..

cmake --build build_openmp --target install --parallel 8

```
Build Kokkos with HIP backend (change CPU and GPU arch as necessary, see https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#cpu-architectures and https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#amd-gpus) *Note: OpenMP and HIP support can be configured in the same build if desired. Note2: modern CMake will possibly auto-detect a different GPU_TARGET, to avoid problems explicitly set the `-DGPU_TARGETS` to your architecture _without_ the `:sramecc+:xnack-` properties.*

```
cmake -B build_hip -S . \
      -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos_HIP -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_ZEN=ON -DKokkos_ARCH_AMD_GFX90A=ON \
      -DGPU_TARGETS="gfx942"

cmake --build build_hip --target install --parallel 8
```

Set Kokkos_DIR to point to external Kokkos package to use

```
export Kokkos_DIR=${HOME}/Kokkos_HIP
```

### Step 2: Modify Build

Get example

```
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
cd Chapter13/Kokkos/StreamTriad
cd Orig
```

Test serial version with 

```
cmake -B build ; cmake --build build ; ./build/StreamTriad
```

If the run fails (SEGV), try reducing the size of the arrays, by reducing the value of the nsize variable in StreamTriad.cc.

Add to CMakeLists.txt

```
(add) find_package(Kokkos REQUIRED)
add_executables(StreamTriad ....)
(add) target_link_libraries(StreamTriad Kokkos::kokkos)
```

Retest with 

```
cmake -B build; cmake --build build
```
and run ./StreamTriad again

Check Ver1 for solution. These modifications have already been made in Ver1 version.

### Step 3: Add Kokkos views for memory allocation of arrays

(peek at ver4/StreamTriad.cc to see the end result)

Add include file

```
#include <Kokkos_Core.hpp>
```

Add initialize and finalize

```
Kokkos::initialize(argc, argv);  {

} Kokkos::finalize();
```

Replace static array declarations with Kokkos views

```
int nsize=80000000;
Kokkos::View<double *> a( "a", nsize);
Kokkos::View<double *> b( "b", nsize);
Kokkos::View<double *> c( "c", nsize);
```

Rebuild and run

```
cmake -B build ; cmake --build build ; ./build/StreamTriad
```

#### Step 4: Add Kokkos execution pattern - parallel_for

Change for loops to Kokkos parallel fors.

At start of loop

```
Kokkos::parallel_for(nsize, KOKKOS_LAMBDA (int i) {
```

  At end of loop, replace closing brace with 

```
});
```

Rebuild and run. Add environment variables as Kokkos message suggests:

```
 export OMP_PROC_BIND=spread
 export OMP_PLACES=threads
 export OMP_NUM_THREADS=8
```

How much speedup do you observe?

### Step 5: Add Kokkos timers

Add Kokkos calls

```
Kokkos::Timer timer;
timer.reset(); // for timer start
time_sum += timer.seconds();
```

Remove

```
#include <timer.h>
struct timespec tstart;
cpu_timer_start(&tstart);
time_sum += cpu_timer_stop(tstart);
```

### 6. Run and measure performance with OpenMP

Find out how many virtual cores are on your CPU

```
lscpu
```

First run with a single processor:

Average runtime ___________

Then run the OpenMP version:

Average runtime ___________

### Portability Exercises

1. Rebuild Stream Triad using Kokkos build with HIP

Set Kokkos_DIR to point to external Kokkos build with HIP

```
export Kokkos_DIR=${HOME}/Kokkos_HIP/lib/cmake/Kokkos_HIP
cmake -B build
cmake --build build
```

2. Run and measure performance with AMD Radeon GPUs

HIP build with ROCm

Ver4 - Average runtime is ______ msecs


