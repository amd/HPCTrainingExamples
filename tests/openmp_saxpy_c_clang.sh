#!/bin/bash

module load clang/15

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C/Make/saxpy

make
./saxpy
make clean
