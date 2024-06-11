#!/bin/bash

module load rocm

cd ${REPO_DIR}/HIPIFY/mini-nbody/cuda
hipify-perl -examine nbody-orig.cu

hipify-perl nbody-orig.cu > nbody-orig.cpp
hipcc -DSHMOO -I../ nbody-orig.cpp -o nbody-orig

./nbody-orig
