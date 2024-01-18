#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran
cd reduction_scalar
make
./reduction_scalar

make clean
