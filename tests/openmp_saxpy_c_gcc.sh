#!/bin/bash

module load amd-gcc/13

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C/saxpy

make
./saxpy
make clean
