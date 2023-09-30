#!/bin/bash

module load rocm

cd ~/HPCTrainingExamples/ManagedMemory/vectorAdd

sed -i 's/\/opt\/rocm/${ROCM_PATH}/g' Makefile

HSA_XNACK=1
make vectoradd_hip2.exe

./vectoradd_hip2.exe
