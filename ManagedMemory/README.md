# Programming Model Exercises -- Managed Memory and Single Address Space (APU)

From `HPCTrainingExamples/ManagedMemory/README.md` in the training exercises repository

**NOTE**: these exercises have been tested on MI210 and MI300A accelerators using a container environment.
To see details on the container environment (such as operating system and modules available) please see `README.md` on [this](https://github.com/amd/HPCTrainingDock) repo.

The source code for these exercises is based on those in the presentation, but with details
filled in so that there is a working code. You may want to examine the code in these exercises
and compare it to the code in the presentation and to the code in the other exercises.

## CPU Code baseline

```
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/ManagedMemory
```
First, run the standard CPU version. This is a working version of the original CPU code from the programming model presentation.
The example will work with any C compiler and run on any CPU. To set up the environment, we need to set the CC environment 
variable to the C compiler executable. We do this by loading the amdclang module which sets `CC=/opt/rocm-<version>/llvm/bin/amdclang`.
The makefile uses the CC environment which we have set. In our modules, we set the "family" to compiler so that only one compiler
can be loaded at a time.

```
cd HPCTrainingExamples/ManagedMemory/CPU_Code
module load amdclang
make
```

will compile with `/opt/rocm-<version>/llvm/bin/amdclang -g -O3 cpu_code.c -o cpu_code`
Then run code with

```
./cpu_code
```

## Standard GPU Code example

This example shows the standard GPU explicit memory management. For this case, we must
move the memory ourselves. This example will run on any AMD Instinct GPU (data center GPUs) and
most workstation or desktop discrete GPUs and APUs. The AMD GPU driver and ROCm software needs
to be installed.

For the environment setup, we need the ROCm bin directory added to the path. We do this by loading the ROCm module with
`module load rocm`. This will set the path to the rocm bin directory. We could also do this with export `PATH=/opt/rocm-<version>/bin`
or by supplying the full path `/opt/rocm-<version>/bin/hipcc` to the compile line. Note that even this may not be
necessary as the ROCm install may have placed a link to `hipcc` in `/usr/bin/hipcc` during the ROCm install. 

We also supply a `--offload-arch=${AMDGPU_GFXMODEL}` option to the compile line. While not necessarily required, it helps
in cases where the architecture is not autodetected properly. We use the following line to query what the 
architecture string `AMDGPU_GFXMODEL` should be. We can also set our own `AMDGPU_GFXMODEL` variable in cases where
we want to cross-compile or compile for more than one architecture.

```
AMDGPU_GFXMODEL ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
```

The `AMDGPU_GFXMODEL` architecture string is `gfx90a` for MI200 series and `gfx942` for MI300A and MI300X. We can
also compile for more than one architecture with `export AMDGPU_GFXMODEL="gfx90a;gfx942"`.

```
cd ../GPU_Code
make
```

This will compile with `hipcc -g -O3 --offload-arch=${AMDGPU_GFXMODEL} gpu_code.hip -o gpu_code`.

Then run the code with

```
./gpu_code
```

## Managed Memory Code

In this example, we will set the `HSA_XNACK` environment variable to 1 and let the Operating System move the memory for us.
This will run on AMD Instinct GPUs for the data center including MI300X, MI300A, and MI200 series. To set up the environment,
`module load rocm`.

```
export HSA_XNACK=1
module load rocm
cd ../Managed_Memory_Code
make
./gpu_code
```

To understand the difference between the explicit memory management programming and the managed memory, let's compare the
two codes.

```
diff gpu_code.hip ../GPU_Code/
```

You should see the following:

```
34a35,37
>    double *in_d, *out_d;
>    HIP_CHECK(hipMalloc((void **)&in_d, Msize));
>    HIP_CHECK(hipMalloc((void **)&out_d, Msize));
38a42,43
>    HIP_CHECK(hipMemcpy(in_d, in_h, Msize, hipMemcpyHostToDevice));
>
41c46
<    gpu_func<<<grid,block,0,0>>>(in_h, out_h, M);
---
>    gpu_func<<<grid,block,0,0>>>(in_d, out_d, M);
43a49
>    HIP_CHECK(hipMemcpy(out_h, out_d, Msize, hipMemcpyDeviceToHost));
```

It may be more instructive to look at the lines of hip code that are required compared to the explicit
memory management GPU code.

```
grep hip ../GPU_Code/gpu_code.hip
```

which gets the following output

```
#include "hip/hip_runtime.h"
    hipError_t gpuErr = call;                            \
    if(hipSuccess != gpuErr){                            \
         __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
   HIP_CHECK(hipMalloc((void **)&in_d, Msize));
   HIP_CHECK(hipMalloc((void **)&out_d, Msize));
   HIP_CHECK(hipMemcpy(in_d, in_h, Msize, hipMemcpyHostToDevice));
   HIP_CHECK(hipDeviceSynchronize());
   HIP_CHECK(hipMemcpy(out_h, out_d, Msize, hipMemcpyDeviceToHost));
```

```
grep hip gpu_code.hip
```

And for the managed memory program, we essentially get just the
addition of the `hipDeviceSynchronize` call plus including the 
hip runtime header and the error checking macro.

```
#include "hip/hip_runtime.h"
    hipError_t gpuErr = call;                            \
    if(hipSuccess != gpuErr){                            \
         __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
   HIP_CHECK(hipDeviceSynchronize());
```

## APU Code -- Single Address Space in HIP

We'll run the same code as we used in the managed memory example. Because 
the memory pointers are addressable on both the CPU and the GPU, no memory management is necessary. First, 
log onto an MI300A node. Then compile and run the code as follows.

```
export HSA_XNACK=1
module load rocm
cd ../APU_Code
make
./gpu_code
```

It may be confusing why we need `HSA_XNACK=1`. Even with the APU, we need to map the pointers into the 
GPU page map though the memory itself does not need to be copied.

## OpenMP APU or single address space

For this example, we have a simple code with the loop offloading in the main code, `openmp_code`, 
and a second version, `openmp_code1`, with the offloaded loop in a subroutine where the compiler 
cannot tell the size of the array. Running this on the MI200 series, it passes, despite that it 
does not have a single address space. We add `export LIBOMPTARGET_INFO=-1` or for less output
`export LIBOMPTARGET_INFO=$((0x1 | 0x10))` to verify that it is running on the GPU.

```
export HSA_XNACK=1
module load amdclang
cd ../OpenMP_Code
make
```

You should see some warnings that are basically telling you the AMD clang compiler is ignoring the `simd` clause
is being ignored. You can remove the `simd` from the OpenMP pragmas, but at the expense of portability to some
other OpenMP compilers. Now run the code.

```
./openmp_code
./openmp_code1
export LIBOMPTARGET_INFO=$((0x1 | 0x10)) # or export LIBOMPTARGET_INFO=-1
./openmp_code
./openmp_code1
```

If the executable is running on the GPU you will see some output as a result of the `LIBOMPTARGET_INFO` environment
variable being set. If it is not running on the GPU, you will not see anything.

For more experimentation with this example, comment out the first line of the two source codes.

```
//#pragma omp requires unified_shared_memory
make
export LIBOMPTARGET_INFO=-1
./openmp_code
./openmp_code1
```

Now with the `LIBOMPTARGET_INFO` variable set, we get a report that memory is being copied to the device
and back. The OpenMP compiler is helping out a lot more than might be expected even without an APU.

## RAJA Single Address Code

First, set up the environment

```
module load amdclang
module load rocm
```

For the Raja example, we need to build the Raja code first

```
cd ~/HPCTrainingExamples/ManagedMemory/Raja_Code

PWDir=`pwd`

git clone --recursive https://github.com/LLNL/RAJA.git Raja_build
cd Raja_build

mkdir build_hip && cd build_hip

cmake -DCMAKE_INSTALL_PREFIX=${PWDir}/Raja_HIP \
      -DROCM_ROOT_DIR=/opt/rocm \
      -DHIP_ROOT_DIR=/opt/rocm \
      -DHIP_PATH=/opt/rocm/bin \
      -DENABLE_TESTS=Off \
      -DENABLE_EXAMPLES=Off \
      -DRAJA_ENABLE_EXERCISES=Off \
      -DENABLE_HIP=On \
      ..

make -j 8
make install

cd ../..

rm -rf Raja_build

export Raja_DIR=${PWDir}/Raja_HIP
```

Now we build the example. Note that we just allocated the arrays on the
host with malloc. To run it on the MI200 series, we need to set the
`HSA_XNACK` variable.

```
# To run with managed memory
export HSA_XNACK=1

mkdir build && cd build
CXX=hipcc cmake ..
make
./raja_code

cd ..
rm -rf build

cd ${PWDir}
rm -rf Raja_HIP

cd ..
rm -rf ${PROB_NAME}
```

## Kokkos Unified Address Code

First, set up the environment

```
module load amdclang
module load rocm
```

For the Kokkos example, we also need to build the Kokkos code first

```
cd ~/HPCTrainingExamples/ManagedMemory/Kokkos_Code

PWDir=`pwd`

git clone https://github.com/kokkos/kokkos Kokkos_build
cd Kokkos_build

mkdir build_hip && cd build_hip
cmake -DCMAKE_INSTALL_PREFIX=${PWDir}/Kokkos_HIP -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_ZEN=ON -DKokkos_ARCH_VEGA90A=ON \
      -DCMAKE_CXX_COMPILER=hipcc ..

make -j 8
make install

cd ../..

rm -rf Kokkos_build

export Kokkos_DIR=${PWDir}/Kokkos_HIP
```

Now we build the example. Note that we have not had to declare the arrays
in Kokkos Views. 

```
# To run with managed memory
export HSA_XNACK=1

mkdir build && cd build
CXX=hipcc cmake ..
make
./kokkos_code

cd ${PWDir}
rm -rf Kokkos_HIP

cd ..
rm -rf ${PROB_NAME}
```

With recent versions of Kokkos, there is support for a single memory copy for the MI300A GPU.

```
-Dkokkos_ENABLE_IMPL_HIP_UNIFIED_MEMORY=ON in Kokkos 4.4+
```

Makes it easy to switch between host/device duplicate arrays to single memory copy on the MI300A.

