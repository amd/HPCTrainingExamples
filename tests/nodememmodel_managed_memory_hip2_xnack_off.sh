#!/bin/bash

module load rocm

cd ${REPO_DIR}/ManagedMemory/vectorAdd

sed -i 's/\/opt\/rocm/${ROCM_PATH}/g' Makefile

export HSA_XNACK=0
make vectoradd_hip2.exe

./vectoradd_hip2.exe
