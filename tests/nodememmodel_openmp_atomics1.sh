#!/bin/bash

module load amdclang

cd ${REPO_DIR}/atomics_openmp

make arraysum1
./arraysum1

make clean

