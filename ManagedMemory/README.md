
--------------------------------------------------------------
\pagebreak{}

# Programming Model Exercises -- Managed Memory and Single Address Space (APU)

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
First run the standard CPU code. This is a working version of the original CPU code from the programming model presentation.

```
cd HPCTrainingExamples/ManagedMemory/CPU_Code
module load amdclang
make
```

will compile with `amdclang -g -O3 -fopenmp cpu_code.c -o cpu_code`
Then run code with

```
./cpu_code
```

## Standard GPU Code example

This example adds the GPU memory corresponding to the CPU arrays and explicitly manages the memory transfers. 

```
cd ../GPU_Code
make
```

This will compile with `hipcc -g -O3 -fopenmp --offload-arch=gfx90a gpu_code.hip -o gpu_code`
Then run the code with

```
./gpu_code
```

## Managed Memory Code

In this example, we will set the `HSA_XNACK` environment variable to 1 and let the Operating System move the memory for us.

```
export HSA_XNACK=1
cd ../Managed_Memory_Code
make
./gpu_code
```

## APU Code -- Single Address Space in HIP

This example is shown on slide 29. We'll run the same code as we used in the managed memory 
example. Because 
the memory pointers are addressable on both the CPU and the GPU, no memory managment is necessary. First, 
log onto an MI300A node. Then compile and run the code as follows.

```
cd ../APU_Code
make
./gpu_code
```

## OpenMP APU or single address space

For this example, we have a simple code with the loop offloading in the main code, `openmp_code`, and a second version, `openmp_code1`, with the offloaded loop in a subroutine where the compiler cannot tell the size of the array. Running this on the MI200 series, it passes, despite that it does not have a single address space. We add `export LIBOMPTARGET_INFO=-1` to verify that it is running on the GPU. 

```
export HSA_XNACK=1
module load amdclang
cd ../OpenMP_Code
make
./openmp_code
./openmp_code1
export LIBOMPTARGET_INFO=-1
./openmp_code
./openmp_code1
```

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

For the Kokkos example, we need to build the Kokkos code first

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
in Kokkos Views. To run it on the MI200 series, we need to set the
`HSA_XNACK` variable.

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

