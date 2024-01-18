#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C
cd reduction_scalar
make
./reduction_scalar

make clean
