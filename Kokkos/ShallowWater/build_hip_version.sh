#!/bin/sh

module purge
module load rocm-new/10.1.0

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_PROC_BIND=true

#if not set by module set it here
#ROCM_PATH=/shared/apps/rhel9/rocm-10.1.0

export CC=${ROCM_PATH}/llvm/bin/amdclang
export CXX=${ROCM_PATH}/llvm/bin/amdclang++

export Kokkos_DIR=${HOME}/install/Kokkos_HIP

mkdir hip_build; cd hip_build
cmake -DCMAKE_PREFIX_PATH=${Kokkos_DIR} -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=${CXX} -DGPU_TARGETS='gfx942' ..
make -j 8

for exec in ShallowWater ShallowWater_par?
do
   echo ""
   echo "Running $exec version"
   ./$exec
   echo "==========================="
done
