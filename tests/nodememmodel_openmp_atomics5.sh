#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/atomics_openmp

make arraysum5
export HSA_XNACK=1
./arraysum5

make clean

