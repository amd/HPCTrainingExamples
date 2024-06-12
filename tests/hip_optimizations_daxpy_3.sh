#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP-Optimizations/daxpy
make daxpy_3
./daxpy_3 1000000

make clean
