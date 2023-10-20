#!/bin/bash

module load amd-gcc

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/freduce

make
./freduce
make clean
