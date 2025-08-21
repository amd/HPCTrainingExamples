#!/bin/bash

module load rocm
module load amdclang

PROB_NAME=kokkos_openmp_stream_orig
mkdir ${PROB_NAME} && cd ${PROB_NAME}

PWDir=`pwd`

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
