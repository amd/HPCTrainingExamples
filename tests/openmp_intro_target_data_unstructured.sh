#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Intro
make target_data_unstructured
./target_data_unstructured

make clean
