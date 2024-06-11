#!/bin/bash

module load rocm

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPIFY/mini-nbody/hip/
make nbody-orig

rocprof --stats --hip-trace ./nbody-orig 65536

cat results.hip_stats.csv

make clean
