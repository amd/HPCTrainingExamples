#!/bin/bash

git clone --recursive https://github.com/LLNL/Umpire.git Umpire_source
cd Umpire_source
sed -i 's/memoryType/type/g' src/umpire/tpl/camp/include/camp/resource/hip.hpp
mkdir -p build && cd build
UMPIRE_PATH=./Umpire_install
mkdir $UMPIRE_PATH

module load rocm
module load gcc

AMDGPU_GFXMODEL=`rocminfo | grep gfx | sed -e 's/Name://' | head -1 |sed 's/ //g'`

cmake -DCMAKE_INSTALL_PREFIX=${UMPIRE_PATH} \
      -DROCM_ROOT_DIR=${ROCM_PATH} \
      -DHIP_ROOT_DIR=${ROCM_PATH}/hip \
      -DHIP_PATH=${ROCM_PATH}/llvm/bin \
      -DENABLE_HIP=On \
      -DENABLE_OPENMP=Off \
      -DENABLE_CUDA=Off \
      -DENABLE_MPI=Off \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_HIP_ARCHITECTURES=$AMDGPU_GFXMODEL \
      -DAMDGPU_TARGETS=$AMDGPU_GFXMODEL \
      -DGPU_TARGETS=$AMDGPU_GFXMODEL \
      -DBLT_CXX_STD=c++14 \
      -DUMPIRE_ENABLE_IPC_SHARED_MEMORY=On \
      ../

make -j 16

make install
