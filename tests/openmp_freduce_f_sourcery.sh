#!/bin/bash

module load sourcery

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/Make/freduce

make
./freduce
make clean
