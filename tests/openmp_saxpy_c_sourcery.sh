#!/bin/bash

module load sourcery

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C/Make/saxpy

make
./saxpy
make clean
