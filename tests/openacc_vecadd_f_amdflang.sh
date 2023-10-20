#!/bin/bash

module load aomp-amdclang

cd ~/HPCTrainingExamples/Pragma_Examples/OpenACC/Fortran/vecadd

make
./vecadd
