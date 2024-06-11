#!/bin/bash

module load amdclang

cd ${REPO_DIR}/atomics_openmp

make arraysum7
export HSA_XNACK=1
./arraysum7

make clean

