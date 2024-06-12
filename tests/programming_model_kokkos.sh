#!/bin/bash

module load amdclang
module load rocm

PROB_NAME=programming_model_kokkos_code
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

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/ManagedMemory/Kokkos_Code

# To run with managed memory
export HSA_XNACK=1

rm -rf build
mkdir build && cd build
CXX=hipcc cmake ..
make
./kokkos_code

cd ..
rm -rf build

cd ${PWDir}
rm -rf Kokkos_HIP Kokkos_build

rm -rf ${PROB_NAME}
