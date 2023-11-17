#!/bin/bash

module load og

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C/saxpy

make
./saxpy
make clean
