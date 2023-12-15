#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/ManagedMemory/APU_Code
export HSA_XNACK=1
make gpu_code
./gpu_code

make clean
