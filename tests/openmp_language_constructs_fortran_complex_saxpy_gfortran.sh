#!/bin/bash

module load amd-gcc
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran
cd complex_saxpy
make
./complex_saxpy

make clean
