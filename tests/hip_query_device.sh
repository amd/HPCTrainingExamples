#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   export HIPCC="`which CC`"
   export HIPFLAGS="-x hip"
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/query_device

make
./query_device
make clean
