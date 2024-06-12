#!/bin/bash

module load rocm

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPIFY/mini-nbody/hip/
make nbody-orig

rocprof --stats ./nbody-orig 65536

grep "bodyForce" results.csv |wc -l

make clean
