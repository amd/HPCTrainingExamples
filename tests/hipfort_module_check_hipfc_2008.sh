#!/bin/bash

module load hipfort
AMDGPU_GFXMODEL=`rocminfo | grep gfx | sed -e 's/Name://' | head -1 |sed 's/ //g'`

git clone https://github.com/ROCm/hipfort hipfort_for_test

pushd hipfort_for_test/test/f2008/vecadd

HIPFORT_COMP=`which amdflang`

# Example with Fortran 2008 interface
hipfc -v --offload-arch=${AMDGPU_GFXMODEL} -hipfort-compiler ${HIPFORT_COMP} hip_implementation.cpp main.f03
./a.out

popd

rm -rf hipfort_for_test

module unload hipfort

