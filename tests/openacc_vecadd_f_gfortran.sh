#!/bin/bash

module load gcc/13

cd ~/HPCTrainingExamples/Pragma_Examples/OpenACC/Fortran/Make/vecadd

make
./vecadd
