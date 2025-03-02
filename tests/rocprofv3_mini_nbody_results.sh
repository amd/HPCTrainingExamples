#!/bin/bash

module load rocm

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPIFY/mini-nbody/hip/
make nbody-orig

mkdir rocprofv3_tests
cd rocprofv3_tests

rocprofv3 --kernel-trace --stats -- ./nbody-orig 65536

cd *

cat *stats.csv

cd ../../

rm -rf rocprofv3_tests

make clean
