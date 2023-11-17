#!/bin/bash

module load clacc

cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C/saxpy

make
./saxpy
make clean
