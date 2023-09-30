#!/bin/bash

module load sourcery

cd ~/HPCTrainingExamples/Pragma_Examples/OpenACC/C/Make/saxpy

make
./saxpy
make clean
