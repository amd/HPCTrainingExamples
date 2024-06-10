#!/bin/bash

module load amdclang

cd ${REPO_DIR}/atomics_openmp

make arraysum5
export HSA_XNACK=1
./arraysum5

make clean

