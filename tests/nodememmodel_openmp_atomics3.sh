#!/bin/sh

module load amdclang

cd ${REPO_DIR}/atomics_openmp

make arraysum3
./arraysum3

make clean

