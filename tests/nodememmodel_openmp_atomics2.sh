#!/bin/sh

module load amdclang

cd ~/HPCTrainingExamples/atomics_openmp

make arraysum2
HSA_XNACK=1
./arraysum2

make clean

