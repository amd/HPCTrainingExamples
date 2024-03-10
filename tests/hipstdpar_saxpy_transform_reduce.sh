#!/bin/bash

export HSA_XNACK=1
module load llvm-latest

cd ~/HPCTrainingExamples/HIPStdPar/CXX/saxpy_transform_reduce

make
export AMD_LOG_LEVEL=3
./saxpy

make clean
