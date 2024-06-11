#!/bin/bash

module load amdclang

cd ${REPO_DIR}/ManagedMemory/CPU_Code
make cpu_code
./cpu_code

make clean
