#!/bin/sh

rm -rf hipfort-source hipfort-build hipfort

# Install HIPFort
ROCM_GPU=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'`
export HIPFORT_INSTALL_DIR=`pwd`/hipfort
git clone https://github.com/ROCmSoftwarePlatform/hipfort hipfort-source
mkdir hipfort-build; cd hipfort-build
cmake -DHIPFORT_INSTALL_DIR=${HIPFORT_INSTALL_DIR} ../hipfort-source
make install            
export PATH=${HIPFORT_INSTALL_DIR}/bin:$PATH

# Try example from source director
cd ../hipfort-source/test/f2003/vecadd
hipfc -v --offload-arch=${ROCM_GPU} hip_implementation.cpp main.f03
./a.out

# Example with Fortran 2008 interface
cd ../../f2008/vecadd
hipfc -v --offload-arch=${ROCM_GPU} hip_implementation.cpp main.f03
./a.out
