#!/bin/bash
module load amdclang

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/CXX/memory_pragmas

export CXX=amdclang++
export LIBOMPTARGET_INFO=-1
export OMP_TARGET_OFFLOAD=MANDATORY

mkdir build && cd build
cmake ..
make mem1
./mem1

cd ..
rm -rf build
