#!/bin/bash

module load rocm
module load amdflang-new >& /dev/null
if [ "$?" == "1" ]; then
   module load amdclang
fi
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/optimization/Allocations/2_opt_allocation

make
./opt_allocation
make clean
