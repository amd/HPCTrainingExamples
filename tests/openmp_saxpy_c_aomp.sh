#!/bin/bash

module load aomp

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C/saxpy

make
./saxpy
make clean
