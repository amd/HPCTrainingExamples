#!/bin/bash

module load clang/15

cd ~/HPCTrainingExamples/Pragma_Examples/OpenACC/Fortran/vecadd

make
./vecadd
