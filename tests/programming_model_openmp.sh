#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/ManagedMemory/OpenMP_Code
make openmp_code
./openmp_code

make clean
