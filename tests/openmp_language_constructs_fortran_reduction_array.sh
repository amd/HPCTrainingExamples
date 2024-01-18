#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran
cd reduction_array
make
./reduction_array

make clean
