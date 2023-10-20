#!/bin/bash

module load sourcery

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/freduce

make
./freduce
make clean
