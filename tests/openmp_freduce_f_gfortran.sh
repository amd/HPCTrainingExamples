#!/bin/bash

module load gcc/13

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/Make/freduce

make
./freduce
make clean
