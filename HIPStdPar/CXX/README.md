# C++ Standard Parallelism on AMD GPUs

Here are some instructions on how to compile and run some tests that exploit C++ standard parallelism.
**NOTE**: these exercises have been tested on MI210 and MI300A accelerators using a container environment.
To see details on the container environment (such as operating system and modules available) please see `README.md` on [this](https://github.com/amd/HPCTrainingDock) repo. 

```
git clone https://github.com/amd/HPCTrainingExamples.git
```

## hipstdpar_saxpy_foreach example

```
export HSA_XNACK=1
module load amdclang

cd ~/HPCTrainingExamples/HIPStdPar/CXX/saxpy_foreach

make
export AMD_LOG_LEVEL=3
./saxpy
clean
```

## hipstdpar_saxpy_transform example

```
export HSA_XNACK=1
module load amdclang

cd ~/HPCTrainingExamples/HIPStdPar/CXX/saxpy_transform

make
export AMD_LOG_LEVEL=3
./saxpy
clean
```

## hipstdpar_saxpy_transform_reduce example

```
export HSA_XNACK=1
module load amdclang

cd ~/HPCTrainingExamples/HIPStdPar/CXX/saxpy_transform_reduce

make
export AMD_LOG_LEVEL=3
./saxpy
clean
```

## Traveling Salesperson Problem

```
#!/bin/bash

git clone https://github.com/pkestene/tsp
cd tsp
git checkout 51587
wget -q https://raw.githubusercontent.com/ROCm/roc-stdpar/main/data/patches/tsp/TSP.patch

patch -p1 < TSP.patch

cd stdpar

export HSA_XNACK=1
module load amdclang
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

## hipstdpar_shallowwater_orig.sh

```
cd ~/HPCTrainingExamples/HIPStdPar/CXX/ShallowWater_Orig

mkdir build && cd build
cmake ..
make
./ShallowWater

cd ..
rm -rf build
```

## hipstdpar_shallowwater_ver1.sh

```
cd ~/HPCTrainingExamples/HIPStdPar/CXX/ShallowWater_Ver1

mkdir build && cd build
cmake ..
make
./ShallowWater

cd ..
rm -rf build
```

## hipstdpar_shallowwater_ver2.sh

```
export HSA_XNACK=1
module load amdclang

cd ~/HPCTrainingExamples/HIPStdPar/CXX/ShallowWater_Ver2

make
#export AMD_LOG_LEVEL=3
./ShallowWater

make clean
```
## hipstdpar_shallowwater_stdpar.sh

```
export HSA_XNACK=1
module load amdclang

cd ~/HPCTrainingExamples/HIPStdPar/CXX/ShallowWater_StdPar

make
#export AMD_LOG_LEVEL=3
./ShallowWater

make clean
```

## Mix and Match

The examples contained in the MixandMatch directory demonstrate how to correctly combine
StdPar with other commonly used programming models, such as OpenMP and HIP.

All examples require the user to specify the path to the StdPar header in the Makefile.

* omp_stdpar: demonstrates how to integrate StdPar and OpenMP within the same application.
It utilizes object-oriented programming techniques to implement the same interface in specialized ways.

* std_cpu_gpu: shows how to combine StdPar sections using `par` and `par_unseq`
to run on both the CPU and GPU within the same application.

* hip_stdpar: illustrates how to use HIP routines to allocate and transfer data to GPU buffers
for use in StdPar sections.

* atomic_stdpar_omp: explains how atomic operations can be safely performed within a StdPar
section using the `par_unseq` policy. The example also includes an equivalent OpenMP implementation.
