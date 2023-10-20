#!/bin/bash

module load aomp

cd HPCTrainingExamples/Pragma_Examples/OpenACC/C/saxpy

make
./saxpy
make clean
