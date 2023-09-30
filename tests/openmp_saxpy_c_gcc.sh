#!/bin/bash

module load gcc/13

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C/Make/saxpy

make
./saxpy
make clean
