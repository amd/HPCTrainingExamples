#!/bin/bash

cd ${REPO_DIR}/HIP-Optimizations/daxpy
make daxpy_5
./daxpy_5 1000000

make clean
