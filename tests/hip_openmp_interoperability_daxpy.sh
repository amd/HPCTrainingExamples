#!/bin/bash
export HSA_XNACK=1
module load amdclang

cd ~/HPCTrainingExamples/HIP-OpenMP/CXX/daxpy
make
./daxpy

make clean
