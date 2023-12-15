#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/ManagedMemory/GPU_Code
make gpu_code
./gpu_code

make clean
