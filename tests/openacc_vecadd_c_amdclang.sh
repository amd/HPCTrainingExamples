#!/bin/bash

module load aomp

cd ~/HPCTrainingExamples/Pragma_Examples/OpenACC/C/Make/vecadd

make
./vecadd
