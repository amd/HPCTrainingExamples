#!/bin/bash

module load clang

cd ~/HPCTrainingExamples/Pragma_Examples/OpenACC/C/Make/vecadd

make
./vecadd
