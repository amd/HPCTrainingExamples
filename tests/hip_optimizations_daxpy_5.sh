#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP-Optimizations/daxpy
module load rocm
make daxpy_5
./daxpy_5 1000000

make clean
