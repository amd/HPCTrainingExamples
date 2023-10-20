#!/bin/bash

module load clang/15

cd HPCTrainingExamples/Pragma_Examples/OpenACC/C/saxpy

make
./saxpy
make clean
