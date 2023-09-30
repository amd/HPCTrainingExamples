#!/bin/bash

module load amdclang

cd ~/HPCTrainingExamples/atomics_openmp

make arraysum8
export HSA_XNACK=1
./arraysum8

make clean

