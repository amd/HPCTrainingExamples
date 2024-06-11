#!/bin/bash

module load rocm

cd ${REPO_DIR}/HIP/hip-stream

make stream
./stream
make clean
