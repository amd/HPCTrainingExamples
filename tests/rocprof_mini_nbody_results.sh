#!/bin/bash

module load rocm

cd ~/HPCTrainingExamples/HIPIFY/mini-nbody/hip/
hipcc -DSHMOO -I ../ nbody-orig.hip -o nbody-orig

rocprof --stats ./nbody-orig 65536

grep "bodyForce" results.csv |wc -l

make clean
