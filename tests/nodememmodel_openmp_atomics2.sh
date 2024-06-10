#!/bin/sh

module load amdclang

cd ${REPO_DIR}/atomics_openmp

make arraysum2
HSA_XNACK=1
./arraysum2

make clean

