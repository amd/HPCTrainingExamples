#!/bin/bash

module load rocm

cd ~/HPCTrainingExamples/HIPIFY/mini-nbody/hip/
make nbody-orig

rocprof --stats ./nbody-orig 65536

cat results.stats.csv

make clean
