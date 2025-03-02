#!/bin/bash

module load rocm

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPIFY/mini-nbody/hip/
make nbody-orig

rocprofv3 --stats --hsa-trace ./nbody-orig 65536

cat results.hsa_stats.csv

make clean
