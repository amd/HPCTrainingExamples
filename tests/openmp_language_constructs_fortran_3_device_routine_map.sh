#!/bin/bash

module load rocm
module load amdflang-new >& /dev/null
if [ "$?" == "1" ]; then
   module load amdclang
fi
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/6_device_routines/device_routine_with_interface/3_device_routine_map
make
./device_routine

make clean
