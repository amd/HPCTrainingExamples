#!/bin/bash

module load amdclang

cd ${REPO_DIR}/atomics_openmp

make arraysum3
export HSA_XNACK=1
./arraysum3

make clean

