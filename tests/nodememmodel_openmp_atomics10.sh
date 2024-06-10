#!/bin/bash

module load amdclang

cd ${REPO_DIR}/atomics_openmp

make arraysum10
export HSA_XNACK=1
./arraysum10

make clean

