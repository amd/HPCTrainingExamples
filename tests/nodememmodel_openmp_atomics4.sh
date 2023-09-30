#!/bin/sh

module load amdclang

cd ~/HPCTrainingExamples/atomics_openmp

make arraysum4
./arraysum4

make clean

