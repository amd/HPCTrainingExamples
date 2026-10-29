#!/bin/sh

module purge
module load rocm-new/10.1.0

#if not set by module set it here
#ROCM_PATH=/shared/apps/rhel9/rocm-10.1.0

export CC=${ROCM_PATH}/llvm/bin/amdclang
export CXX=${ROCM_PATH}/llvm/bin/amdclang++

export Kokkos_DIR=${HOME}/install/Kokkos_HIP

mkdir serial_build; cd serial_build
cmake -DCMAKE_PREFIX_PATH=${Kokkos_DIR} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_CXX_FLAGS=-DFORCE_SERIAL \
      ..
make -j 8

# binaries land in the repo root (CMAKE_SOURCE_DIR), not in this build dir
cd ..

for exec in ShallowWater ShallowWater_par?
do
   echo ""
   echo "Running $exec version"
   ./$exec
   echo "==========================="
done
