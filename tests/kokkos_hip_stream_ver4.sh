#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   module switch PrgEnv-cray PrgEnv-amd
   export CXX=${ROCM_PATH}/llvm/bin/amdclang++
else
   module load amdclang
fi
module load kokkos

rm -rf Chapter13
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
pushd Chapter13/Kokkos/StreamTriad/Ver4
sed -i -e 's/80000000/100000/' StreamTriad.cc

rm -rf build
mkdir build && cd build
CXX=hipcc cmake ..
make
./StreamTriad

popd
rm -rf Chapter13
