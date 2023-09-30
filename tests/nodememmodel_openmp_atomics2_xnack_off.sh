#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/atomics_openmp

make arraysum2
export HSA_XNACK=0
./arraysum2

make clean

