#!/bin/bash

module load amdclang
module load rocm

PROB_NAME=kokkos_hip_stream_ver4
mkdir ${PROB_NAME} && cd ${PROB_NAME}

PWDir=`pwd`

git clone https://github.com/kokkos/kokkos Kokkos_build
cd Kokkos_build

rm -rf build_hip
mkdir build_hip && cd build_hip
cmake -DCMAKE_INSTALL_PREFIX=${PWDir}/Kokkos_HIP -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_ZEN=ON -DKokkos_ARCH_VEGA90A=ON \
      -DCMAKE_CXX_COMPILER=hipcc ..

make -j 8
make install

cd ../..

rm -rf Kokkos_build

export Kokkos_DIR=${PWDir}/Kokkos_HIP

git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
cd Chapter13/Kokkos/StreamTriad/Ver1
sed -i -e 's/80000000/100000/' StreamTriad.cc

rm -rf build
mkdir build && cd build
CXX=hipcc cmake ..
make
./StreamTriad

cd ${PWDir}
rm -rf Kokkos_HIP
rm -rf Chapter13

cd ..
rm -rf ${PROB_NAME}
