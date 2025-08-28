#!/bin/bash

module load rocm
module load amdclang 
module load hipfort
AMDGPU_GFXMODEL=`rocminfo | grep gfx | sed -e 's/Name://' | head -1 |sed 's/ //g'`

git clone https://github.com/ROCm/hipfort hipfort_for_test_rocm_2003

pushd hipfort_for_test_rocm_2003/test/f2003/vecadd

HIPFORT_COMP=`which amdflang`

# Try example from source director
hipfc -v --offload-arch=${AMDGPU_GFXMODEL} -hipfort-compiler $HIPFORT_COMP  hip_implementation.cpp main.f03
./a.out

popd

rm -rf hipfort_for_test_rocm_2003

module unload hipfort

