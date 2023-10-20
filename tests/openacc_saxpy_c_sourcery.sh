#!/bin/bash

module load sourcery

cd ~/HPCTrainingExamples/Pragma_Examples/OpenACC/C/saxpy

make
./saxpy
make clean
