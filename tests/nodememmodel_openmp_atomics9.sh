#!/bin/bash

module load amdclang

cd ${REPO_DIR}/atomics_openmp

make arraysum9
export HSA_XNACK=1
./arraysum9

make clean

