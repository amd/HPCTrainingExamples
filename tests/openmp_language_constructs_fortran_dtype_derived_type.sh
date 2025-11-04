#!/bin/bash

module load rocm
module load amdflang-new >& /dev/null
if [ "$?" == "1" ]; then
   module load amdclang
fi
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran
cd 6_derived_types
export HSA_XNACK=1
make dtype_derived_type
./dtype_derived_type

make clean
