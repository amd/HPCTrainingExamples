#!/bin/bash

module load amd-clang

cd HPCTrainingExamples/Pragma_Examples/OpenACC/C/saxpy

make
./saxpy
make clean
