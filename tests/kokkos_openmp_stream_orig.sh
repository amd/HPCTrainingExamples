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

PROB_NAME=kokkos_openmp_stream_orig
mkdir ${PROB_NAME} && cd ${PROB_NAME}

PWDir=`pwd`

rm -rf Chapter13
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
cd Chapter13/Kokkos/StreamTriad/Orig
sed -i -e 's/80000000/100000/' StreamTriad.cc

rm -rf build
mkdir build && cd build
cmake ..
make
./StreamTriad

cd ${PWDir}
rm -rf Chapter13

cd ..
rm -rf ${PROB_NAME}
