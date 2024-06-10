#!/bin/bash

module load rocm

cd ${REPO_DIR}/HIP/saxpy

make saxpy
./saxpy
make clean
