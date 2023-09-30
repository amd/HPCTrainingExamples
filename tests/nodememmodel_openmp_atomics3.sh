#!/bin/sh

module load amdclang

cd ~/HPCTrainingExamples/atomics_openmp

make arraysum3
./arraysum3

make clean

