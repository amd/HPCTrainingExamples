#!/bin/bash

module load rocm

cd ${REPO_DIR}/ManagedMemory/vectorAdd

sed -i 's/\/opt\/rocm/${ROCM_PATH}/g' Makefile

make vectoradd_hip4.exe

./vectoradd_hip4.exe

make clean
