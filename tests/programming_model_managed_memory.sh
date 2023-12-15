#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/ManagedMemory/Managed_Memory_Code
export HSA_XNACK=1
make gpu_code
./gpu_code

make clean
