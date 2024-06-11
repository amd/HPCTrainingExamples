#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP-Optimizations/daxpy
make daxpy_4
./daxpy_4 1000000

make clean
