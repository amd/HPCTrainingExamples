#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load amdclang

PROB_NAME=kokkos_hip_stream_ver2
mkdir ${PROB_NAME} && cd ${PROB_NAME}

PWDir=`pwd`

git clone --branch 4.5.01 https://github.com/kokkos/kokkos Kokkos_build
cd Kokkos_build
sed -i '194d' core/src/HIP/Kokkos_HIP_KernelLaunch.hpp

rm -rf build_hip
mkdir build_hip && cd build_hip
cmake -DCMAKE_INSTALL_PREFIX=${PWDir}/Kokkos_HIP \
      -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ENABLE_HIP=ON \
      -DKokkos_ARCH_AMD_GFX942_APU=ON \
      -DKokkos_ARCH_ZEN=ON \
      -DCMAKE_CXX_COMPILER=hipcc ..

make
make install

cd ../..

rm -rf Kokkos_build

export Kokkos_DIR=${PWDir}/Kokkos_HIP

rm -rf Chapter13
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
cd Chapter13/Kokkos/StreamTriad/Ver2
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
