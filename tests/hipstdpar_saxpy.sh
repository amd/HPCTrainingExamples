#!/bin/bash

export HSA_XNACK=1
module load llvm-latest

cd ~/HPCTrainingExamples/HIPStdPar/CXX/saxpy

make
./saxpy

make clean
