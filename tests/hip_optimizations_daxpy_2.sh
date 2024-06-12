#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP-Optimizations/daxpy
make daxpy_2
./daxpy_2 1000000

make clean
