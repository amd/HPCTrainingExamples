#!/bin/sh

module purge
module load rocm-new/10.1.0

export OMP_PROC_BIND=true
export OMP_PLACES=cores

# fall back to the default install if the module did not set this
: ${ROCM_PATH:=/shared/apps/rhel9/rocm-10.1.0}
export ROCM_PATH

export CC=${ROCM_PATH}/llvm/bin/amdclang
export CXX=${ROCM_PATH}/llvm/bin/amdclang++

: ${Kokkos_DIR:=${HOME}/install/Kokkos_HIP}
export Kokkos_DIR

mkdir -p hip_build; cd hip_build
cmake -DCMAKE_PREFIX_PATH=${Kokkos_DIR} -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=${CXX} -DGPU_TARGETS='gfx942' ..
make -j 8

# binaries land in the repo root (CMAKE_SOURCE_DIR), not in this build dir
cd ..

# set only now, not during the build: ROCm CMake appends a :sramecc+:xnack+
# suffix to the GPU arch at compile time if this is set earlier, which
# conflicts with the plain gfx942 arch Kokkos was built with
export HSA_XNACK=1

for exec in ShallowWater ShallowWater_par?
do
   echo ""
   echo "Running $exec version"
   ./$exec
   echo "==========================="
done
