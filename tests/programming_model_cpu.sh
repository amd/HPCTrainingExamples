#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/ManagedMemory/CPU_Code
make cpu_code
./cpu_code

make clean
