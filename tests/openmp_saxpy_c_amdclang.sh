#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C/saxpy

make
./saxpy
make clean
