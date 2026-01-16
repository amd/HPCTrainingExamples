#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load amdclang
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/optimization/Allocations/3_memorypool

./umpire_setup.sh
export UMPIRE_PATH=${PWD}/Umpire_install
make
./memorypool
make clean

rm -rf Umpire_source Umpire_install
