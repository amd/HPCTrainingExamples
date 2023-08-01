#!/bin/sh
unset LIBOMPTARGET_INFO
unset LIBOMPTARGET_KERNEL_TRACE
unset OMP_TARGET_OFFLOAD

export CXX=amdclang++
export LIBOMPTARGET_INFO=$((0x01 | 0x02 | 0x04 | 0x08 | 0x10 | 0x20))
export OMP_TARGET_OFFLOAD=MANDATORY

#export LIBOMPTARGET_KERNEL_TRACE=2

rm -rf build
mkdir build && cd build
#cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake ..
make VERBOSE=1
#rocgdb -tui ./memory_pragmas
#rocgdb ./memory_pragmas 10000
./mem1 10000 #>& mem1_run.out
./mem2 10000 #>& mem2_run.out
./mem3 10000 #>& mem3_run.out
./mem4 10000 #>& mem4_run.out
./mem5 10000 #>& mem5_run.out
./mem6 10000 #>& mem6_run.out
export HSA_XNACK=1
./mem7 10000 #>& mem7_run.out
./mem8 10000 #>& mem8_run.out
./mem9 10000 #>& mem9_run.out
./mem10 10000 #>& mem10_run.out
./mem11 10000 #>& mem11_run.out
