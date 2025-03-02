#!/bin/bash

module load rocm

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPIFY/mini-nbody/hip/
make nbody-orig

rocprofv3 --stats --basenames on ./nbody-orig 65536

cat results.stats.csv

make clean
