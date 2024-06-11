#!/bin/bash

module load amdclang

cd ${REPO_DIR}/Pragma_Examples/OpenMP/USM/vector_add_usm
make
make run

make clean
