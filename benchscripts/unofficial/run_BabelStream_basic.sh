#!/bin/bash

#echo "Removing old version if it it exists"
rm -rf BabelStream
#echo "Getting apu branch version from https://github.com/UoB-HPC/BabelStream"
git clone https://github.com/UoB-HPC/BabelStream.git
#echo "Building HPL code"
module load rocm amdclang openmpi
cd BabelStream/
cmake -Bbuild -H. -DMODEL=hip -DCXX_EXTRA_FLAGS=-D__HIP_PLATFORM_AMD__ -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc
cmake --build build
# Running HPL
build/hip-stream |& tee BableStream.out


