#!/bin/bash

module load rocm
module load amdflang-new
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/optimization/Allocations/3_memorypool

./umpire_setup.sh
export UMPIRE_PATH=${PWD}/Umpire_install
make
./memorypool
make clean

rm -rf Umpire_source Umpire_install
