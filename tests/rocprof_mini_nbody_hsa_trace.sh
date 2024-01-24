#!/bin/bash

module load rocm

cd ~/HPCTrainingExamples/HIPIFY/mini-nbody/hip/
hipcc -DSHMOO -I ../ nbody-orig.hip -o nbody-orig

rocprof --stats --hsa-trace ./nbody-orig 65536

cat results.hsa_stats.csv

make clean
