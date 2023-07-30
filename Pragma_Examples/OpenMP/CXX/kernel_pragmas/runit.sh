#!/bin/sh
unset LIBOMPTARGET_INFO
unset LIBOMPTARGET_KERNEL_TRACE
unset OMP_TARGET_OFFLOAD

export CXX=amdclang++
export LIBOMPTARGET_KERNEL_TRACE=1
export HSA_XNACK=1
export OMP_TARGET_OFFLOAD=MANDATORY

rm -rf build
mkdir build && cd build
cmake ..
make VERBOSE=1
./kernel1 10000 >& kernel1_run.out
./kernel2 10000 >& kernel2_run.out
./kernel3 10000 >& kernel3_run.out
