#!/bin/sh

module load amdclang

cd ${REPO_DIR}/atomics_openmp

make arraysum4
./arraysum4

make clean

