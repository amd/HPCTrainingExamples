#!/bin/bash

module load gcc/13

cd ~/HPCTrainingExamples/Pragma_Examples/OpenACC/C/Make/saxpy

make
./saxpy
make clean
