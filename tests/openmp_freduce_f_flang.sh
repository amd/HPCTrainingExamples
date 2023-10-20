#!/bin/bash

module load clang/15

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/freduce

make
./freduce
make clean
