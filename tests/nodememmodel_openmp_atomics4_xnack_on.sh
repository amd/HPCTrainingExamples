#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/atomics_openmp

make arraysum4
export HSA_XNACK=1
./arraysum4

make clean

