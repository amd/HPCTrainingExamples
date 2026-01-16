#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPIFY/mini-nbody/hip/
make nbody-orig

mkdir rocprofv3_tests
cd rocprofv3_tests

rocprofv3 --stats --sys-trace -- ../nbody-orig 65536

cd *

cat *

cd ../../

rm -rf rocprofv3_tests

make clean
