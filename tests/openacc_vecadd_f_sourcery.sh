#!/bin/bash

module load sourcery/2022-09.7

cd ~/HPCTrainingExamples/Pragma_Examples/OpenACC/Fortran/vecadd

make
./vecadd
