#!/bin/bash

module load clang

cd ~/HPCTrainingExamples/Pragma_Examples/OpenACC/C/vecadd

make
./vecadd
