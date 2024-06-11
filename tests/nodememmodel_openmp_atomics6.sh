#!/bin/bash

module load amdclang

cd ${REPO_DIR}/atomics_openmp

make arraysum6
export HSA_XNACK=1
./arraysum6

make clean

