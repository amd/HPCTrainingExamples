#!/bin/bash
set -e

UMPIRE_PATH=${PWD}/Umpire_install
rm -rf Umpire_source
for attempt in 1 2 3; do
  if git clone --recursive --depth 1 --shallow-submodules \
       https://github.com/LLNL/Umpire.git Umpire_source; then
    break
  fi
  echo "git clone attempt $attempt failed, retrying in 5s..."
  rm -rf Umpire_source
  sleep 5
done
cd Umpire_source
sed -i 's/memoryType/type/g' src/tpl/umpire/camp/include/camp/resource/hip.hpp 2>/dev/null || true
sed -i 's/Mfree/ffree-form/g' examples/cookbook/CMakeLists.txt 2>/dev/null || true
sed -i 's/Mfree/ffree-form/g' examples/tutorial/fortran/CMakeLists.txt 2>/dev/null || true
sed -i 's/Mfree/ffree-form/g' src/umpire/interface/c_fortran/CMakeLists.txt 2>/dev/null || true
sed -i 's/Mfree/ffree-form/g' tests/integration/interface/fortran/CMakeLists.txt 2>/dev/null || true
mkdir -p build && cd build
mkdir -p $UMPIRE_PATH

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load amdflang-new >& /dev/null
if [ "$?" == "1" ]; then
   module load amdclang
fi

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
      -DCMAKE_Fortran_COMPILER=$FC \
      -DCMAKE_HIP_ARCHITECTURES=$AMDGPU_GFXMODEL \
      -DAMDGPU_TARGETS=$AMDGPU_GFXMODEL \
      -DGPU_TARGETS=$AMDGPU_GFXMODEL \
      -DBLT_CXX_STD=c++20 \
      -DUMPIRE_ENABLE_IPC_SHARED_MEMORY=On \
      -DENABLE_FORTRAN=On \
      ../

make -j 16

make install
