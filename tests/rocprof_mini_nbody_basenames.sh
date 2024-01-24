#!/bin/bash

module load rocm

cd ~/HPCTrainingExamples/HIPIFY/mini-nbody/hip/
hipcc -DSHMOO -I ../ nbody-orig.hip -o nbody-orig

rocprof --stats --basenames on ./nbody-orig 65536

cat results.stats.csv

make clean
rm -f results.copy_stats.csv results.csv results.db results.hip_stats.csv
rm -f results.hsa_stats.csv results.json results.stats.csv results.sysinfo.txt
