#!/bin/bash

cd ${REPO_DIR}/HIP-Optimizations/daxpy
make daxpy_1
./daxpy_1 1000000

make clean
