#!/bin/bash

export HSA_XNACK=1
module load llvm-latest-gcc11

cd ~/HPCTrainingExamples/HIPStdPar/CXX/saxpy

make
./saxpy

make clean
