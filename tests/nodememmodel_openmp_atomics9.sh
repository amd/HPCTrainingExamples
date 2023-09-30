#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/atomics_openmp

make arraysum9
export HSA_XNACK=1
./arraysum9

make clean

