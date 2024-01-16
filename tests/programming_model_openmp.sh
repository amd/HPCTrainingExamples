#!/bin/bash
export HSA_XNACK=1

module load amdclang

cd ~/HPCTrainingExamples/ManagedMemory/OpenMP_Code
make openmp_code
./openmp_code

make clean
