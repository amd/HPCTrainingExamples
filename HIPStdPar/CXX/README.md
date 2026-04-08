
# C++ Standard Parallelism on AMD GPUs

Here are some instructions on how to compile and run some tests that exploit C++ standard parallelism, which is available with ROCm, starting from version 6.1.1. Hence, please double check the version of ROCm you are using to make sure it has HIPSTDPAR enabled. HIPSTDPAR relies on the LLVM compiler, the hipstdpar header only library, and rocThrust.

**NOTE**: these exercises have been tested on MI210 and MI300A accelerators using a container environment and with rocm 7.2 on aac6 and aac7.
To see details on the container environment (such as operating system and modules available) please see `README.md` on [this](https://github.com/amd/HPCTrainingDock) repo. 

```
git clone https://github.com/amd/HPCTrainingExamples.git
```
Allocate a compute node the recommended way on the system you are using.

```
export HSA_XNACK=1
```
Depending on the system you are working on:
```
module load rocm #e.g. on AAC6
```
check if CXX is set to amdclang++

```
echo $CXX
```
on AAC7 for example it is not set, so there one needs to set it manually:
```
module load rocm-new #on AAC7
export CXX=amdclang++

```

## hipstdpar_saxpy_foreach example

```
cd ~/HPCTrainingExamples/HIPStdPar/CXX/saxpy_foreach

```
run the example:
```
make
./saxpy
```
But did it run on the device? With setting
```
export AMD_LOG_LEVEL=3
```
and running again
```
./saxpy
```
you can see some information on what is running on the GPU:
```
:3:rocdevice.cpp            :415 : 578897276813 us:  Initalizing runtime stack, Enumerated GPU agents = 1
:3:rocdevice.cpp            :182 : 578897276884 us:  Numa selects cpu agent[0]=0xfacbd0(fine=0x1031370,coarse=0x1032cd0) for gpu agent=0x106cee0 CPU<->GPU XGMI=1
:3:rocsettings.cpp          :269 : 578897276902 us:  Using dev kernel arg wa = 1
:3:comgrctx.cpp             :126 : 578897276947 us:  Loaded COMGR library version 3.0.
:3:rocdevice.cpp            :1565: 578897277852 us:  addressableNumVGPRs=512, totalNumVGPRs=512, vGPRAllocGranule=8, availableRegistersPerCU_=131072
:3:rocdevice.cpp            :1579: 578897277867 us:  imageSupport=0
:3:rocdevice.cpp            :1610: 578897277874 us:  Gfx Major/Minor/Stepping: 9/4/2
:3:rocdevice.cpp            :1612: 578897277878 us:  HMM support: 1, XNACK: 1, Direct host access: 1
:3:rocdevice.cpp            :1614: 578897277881 us:  Max SDMA Read Mask: 0xfff, Max SDMA Write Mask: 0xfff
:3:hip_context.cpp          :60  : 578897280595 us:  HIP Version: 7.2.26015.fc0010cf6a, Direct Dispatch: 1
:3:os_posix.cpp             :934 : 578897280611 us:  HIP Library Path: /shared/apps/rhel9/rocm-7.2.0/lib/libamdhip64.so.7
:3:hip_platform.cpp         :255 : 578897280660 us:   __hipPushCallConfiguration ( {4,1,1}, {256,1,1}, 0, char array:<null> )
:3:hip_platform.cpp         :259 : 578897280667 us:  __hipPushCallConfiguration: Returned hipSuccess :
:3:hip_platform.cpp         :264 : 578897280679 us:   __hipPopCallConfiguration ( {16434304,0,4294966880}, {1337866000,32765,3067051520}, 0x7ffd4fbe3668, 0x7ffd4fbe3660 )
:3:hip_platform.cpp         :273 : 578897280684 us:  __hipPopCallConfiguration: Returned hipSuccess :
:3:hip_module.cpp           :825 : 578897280720 us:   hipLaunchKernel ( 0x201a40, {4,1,1}, {256,1,1}, 0x7ffd4fbe36a0, 0, char array:<null> )
:3:hip_fatbin.cpp           :524 : 578897280855 us:  Forcing SPIRV: false
:3:hip_fatbin.cpp           :538 : 578897280863 us:  Using native code object for device: amdgcn-amd-amdhsa--gfx942:sramecc+:xnack+ co: amdgcn-amd-amdhsa--gfx942:sramecc+:xnack+
:3:rocdevice.cpp            :2870: 578897322727 us:  Number of allocated hardware queues with low priority: 0, with normal priority: 0, with high priority: 0, maximum per priority is: 4
:3:rocdevice.cpp            :2951: 578897353351 us:  Created SWq=0x7f8ec8e86000 to map on HWq=0x7f8eb5c00000 with size 16384 with priority 1, cooperative: 0
:3:rocdevice.cpp            :3045: 578897353425 us:  acquireQueue refCount: 0x7f8eb5c00000 (1)
:3:rocvirtual.cpp           :3596: 578897390463 us:  ShaderName : void thrust::THRUST_200805_400200_NS::hip_rocprim::__parallel_for::kernel<256u, thrust::THRUST_200805_400200_NS::hip_rocprim::for_each_f<__gnu_cxx::__normal_iterator<double*, std::vector<double, std::allocator<double> > >, thrust::THRUST_200805_400200_NS::detail::wrapped_function<main::{lambda(double&)#1}, void> >, long, 1u>(thrust::THRUST_200805_400200_NS::hip_rocprim::for_each_f<__gnu_cxx::__normal_iterator<double*, std::vector<double, std::allocator<double> > >, thrust::THRUST_200805_400200_NS::detail::wrapped_function<main::{lambda(double&)#1}, void> >, long, long)
:3:hip_module.cpp           :826 : 578897390507 us:  hipLaunchKernel: Returned hipSuccess : : duration: 109787 us
:3:hip_error.cpp            :41  : 578897390521 us:   hipPeekAtLastError (  )
:3:hip_error.cpp            :43  : 578897390525 us:  hipPeekAtLastError: Returned hipSuccess :
:3:hip_error.cpp            :34  : 578897390530 us:   hipGetLastError (  )
:3:hip_stream.cpp           :403 : 578897390538 us:   hipStreamSynchronize ( char array:<null> )
:3:hip_stream.cpp           :404 : 578897390765 us:  hipStreamSynchronize: Returned hipSuccess :
:3:hip_error.cpp            :34  : 578897390771 us:   hipGetLastError (  )
Finished Run - x[10]: 5
:3:rocdevice.cpp            :3077: 578897391176 us:  releaseQueue refCount:0x7f8eb5c00000 (0)
:1:rocdevice.cpp            :3339: 578897397477 us:  Unknown Event Type

```
Note: if you have more than 1 GPU on a node available you may see more than one device shown in Enumerated GPU agents=<num>. You can set ROCR_VISIBLE_DEVICES=0 and HIP_VISIBLE_DEVICES=0 to limit the visibility to one GPU for example. You will see multiple kernels running on all available GPUs otherwise. It depends on your application what you intend, so be careful with setting CPU and GPU  affinity!

## hipstdpar_saxpy_transform example

Make sure you have the amdclang++ compiler loaded and HSA_XNACK=1 set.
```
cd ~/HPCTrainingExamples/HIPStdPar/CXX/saxpy_transform

make
./saxpy
make clean
```
you may also want to repeat with setting AMD_LOG_LEVEL to see what is running on the GPU.

## hipstdpar_saxpy_transform_reduce example
Make sure you have the amdclang++ compiler loaded and HSA_XNACK=1 set.

```
cd ~/HPCTrainingExamples/HIPStdPar/CXX/saxpy_transform_reduce

make
./saxpy
make clean
```
you may also want to repeat with setting AMD_LOG_LEVEL to see what is running on the GPU.

## Traveling Salesperson Problem
Make sure you have the amdclang++ compiler loaded and HSA_XNACK=1 set.

```
#!/bin/bash

git clone https://github.com/pkestene/tsp
cd tsp
git checkout 51587
wget -q https://raw.githubusercontent.com/ROCm/roc-stdpar/main/data/patches/tsp/TSP.patch

patch -p1 < TSP.patch

cd stdpar

export STDPAR_CXX=$CXX
export ROCM_GPU=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'`
export STDPAR_TARGET=${ROCM_GPU}

export AMD_LOG_LEVEL=3 #optional

make tsp_clang_stdpar_gpu
./tsp_clang_stdpar_gpu 13 #or more...

make clean
cd ../..
rm -rf tsp
```

## Shallow Water example of porting to hipstdpar (overview)

| Directory | Language / build | Parallelism |
|-----------|------------------|-------------|
| `ShallowWater_Orig` | C + CMake | Sequential CPU |
| `ShallowWater_Ver1` | C++ + CMake | Sequential CPU; small grid + DEBUG |
| `ShallowWater_Ver2` | C++ + **make** (`--hipstdpar`) | `std::execution::par_unseq`; small grid + DEBUG |
| `ShallowWater_StdPar` | C++ + **make** (`--hipstdpar`) | Same as Ver2; **full** grid, no DEBUG |

### Shallow Water Original version

This example is the **original sequential CPU implementation in C** (`ShallowWater.c`, plus `memory.c` and `timer.c`). It does not use C++ standard parallelism or the GPU.

You only need a C compiler and CMake; a ROCm installation is **not** required to build and run this variant, here will will use amdclang, but any C compiler will do:

```
export CC=amdclang
cd ~/HPCTrainingExamples/HIPStdPar/CXX/ShallowWater_Orig

mkdir build && cd build
cmake ..
make
./ShallowWater

cd ..
rm -rf build
```

If you set `AMD_LOG_LEVEL`, you will not see GPU activity, because this binary does not launch GPU kernels.

### Shallow Water Version 1: Port to C++ as first step for porting to stdpar

This step ports the algorithm to **C++** using a small **`Var2D`** helper (`Var2D.hpp`) and keeps the **same sequential nested loops** as the original (no `std::execution`, no GPU offload). The problem size is **small** (`nx=6`, `ny=4`, few time steps) and **DEBUG output** is enabled so you can trace values.

Any **C++17** toolchain is sufficient; this CMake build does **not** pass `--hipstdpar`. Setting `AMD_LOG_LEVEL` **will not** show GPU kernels for this executable. Her we will use amdclang++, but any C++ compiler will do:

```
export CXX=amdclang++
cd ~/HPCTrainingExamples/HIPStdPar/CXX/ShallowWater_Ver1

mkdir build && cd build
cmake ..
make
export AMD_LOG_LEVEL=3 #optional
./ShallowWater

cd ..
rm -rf build
```

### Shallow Water Version 2: Port to stdpar

This variant adds **`Range2D` and a `range` iterator** in `Var2D.hpp`, then rewrites the hot loops using **`std::for_each`** and **`std::transform_reduce`** with **`std::execution::par_unseq`**. It still uses a **small grid** and **DEBUG** output for validation.

Build with **`make`** (not CMake). The Makefile enables **`--hipstdpar`** (and GPU arch) when using **`amdclang++`** or **`clang++`**.

Make sure you have the amdclang++ compiler loaded and HSA_XNACK=1 set.

```
export CXX=amdclang++
cd ~/HPCTrainingExamples/HIPStdPar/CXX/ShallowWater_Ver2

make
export AMD_LOG_LEVEL=3 #optional
./ShallowWater

make clean
```

### Shallow Water hipstdpar

Same **parallel algorithms and policies** as Version 2 (`par_unseq`, `for_each`, `transform_reduce`), but with the **full problem size** from the original C example and **without DEBUG**.

Use **`amdclang++`**, **`HSA_XNACK=1`** and build with **`make`** as in Version 2.

```
export CXX=amdclang++
cd ~/HPCTrainingExamples/HIPStdPar/CXX/ShallowWater_StdPar

make
export AMD_LOG_LEVEL=3 #optional
./ShallowWater

make clean
```

## Mix and Match

The examples contained in the `MixAndMatch` directory demonstrate how to correctly combine
StdPar with other commonly used programming models, such as OpenMP and HIP.

All examples require the user to specify the path to the StdPar header in the Makefile:

```
module load rocm
export STDPAR_PATH=${ROCM_PATH}/include/thrust/system/hip/hipstdpar
export HSA_XNACK=1
```

Note HIPSTDPAR assumes the device is HMM enabled and setting `HSA_XNACK=1` to one is required. In devices where HMM is not enabled, the additional compilation flag `--hipstdpar-interpose-alloc` needs to be included. This will instruct the compiler to replace all dynamic memory allocations with compatible `hipManagedMemory` allocations.


* omp_stdpar: demonstrates how to integrate StdPar and OpenMP within the same application.
It utilizes object-oriented programming techniques to implement the same interface in specialized ways.

* std_cpu_gpu: shows how to combine StdPar sections using `par` and `par_unseq`
to run on both the CPU and GPU within the same application.

* hip_stdpar: illustrates how to use HIP routines to allocate and transfer data to GPU buffers
for use in StdPar sections. You can use `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_PATH/lib/rocprofiler-systems` for an installation of TBB (needed for stdpar on CPU). The Makefile also uses this TBB version shipped with rocprofiler-systems (works with rocm 7.2, not with rocm 6.4). Make sure to set TBB_LIBDIR if you want to use another TBB version.

* atomic_stdpar_omp: explains how atomic operations can be safely performed within a StdPar
section using the `par_unseq` policy. The example also includes an equivalent OpenMP implementation.

