#!/bin/bash
module load amdclang

cd ~/HPCTrainingExamples/HIP-OpenMP/CXX/daxpy
make
./daxpy

make clean
