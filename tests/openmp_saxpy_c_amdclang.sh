#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C/Make/saxpy

make
./saxpy
make clean
