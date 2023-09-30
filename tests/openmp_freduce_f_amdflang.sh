#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/Make/freduce

make
./freduce
make clean
