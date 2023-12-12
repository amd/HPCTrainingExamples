#!/bin/bash

module load gcc/13

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/vecadd

make
./vecadd
make clean
