#!/bin/bash

module load rocm

cd ${REPO_DIR}/HIPIFY/mini-nbody/hip/
make nbody-orig

rocprof --stats --hsa-trace ./nbody-orig 65536

cat results.hsa_stats.csv

make clean
