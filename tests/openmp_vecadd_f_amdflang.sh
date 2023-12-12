#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/vecadd

make
./vecadd
make clean
