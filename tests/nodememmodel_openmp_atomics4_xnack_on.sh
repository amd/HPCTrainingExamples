#!/bin/bash

module load amdclang

cd ${REPO_DIR}/atomics_openmp

make arraysum4
export HSA_XNACK=1
./arraysum4

make clean

