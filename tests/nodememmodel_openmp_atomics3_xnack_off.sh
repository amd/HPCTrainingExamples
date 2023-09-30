#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/atomics_openmp

make arraysum3
export HSA_XNACK=0
./arraysum3

make clean

