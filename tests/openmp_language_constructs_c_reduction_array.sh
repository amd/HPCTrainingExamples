#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C
cd reduction_array
make
./reduction_array

make clean
