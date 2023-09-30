#!/bin/bash

module load aomp

cd ~/HPCTrainingExamples/Pragma_Examples/OpenACC/Fortran/Make/vecadd

make
./vecadd
