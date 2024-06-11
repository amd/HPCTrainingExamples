#!/bin/bash

module load rocm

cd ${REPO_DIR}/HIP/vectorAdd

make vectoradd
./vectoradd
make clean
