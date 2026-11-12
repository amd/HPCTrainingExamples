#!/bin/bash

# LCOMPILER-2959: amdflang miscompiles at -O0 when a `parallel do` is nested in
# a sequential do loop inside `target teams distribute` (wrong results or a GPU
# memory access fault). -O1 and above are unaffected. Regression since 7.1.1.

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   [ -z "$FC" ] && export FC=`which ftn`
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
fi
if [ -z "$FC" ]; then
   for c in amdflang-new amdflang flang-new flang; do
      command -v $c >/dev/null 2>&1 && { FC=$c; break; }
   done
fi

XNACK_COUNT=`rocminfo | grep xnack | wc -l`
if [ ${XNACK_COUNT} -lt 1 ]; then
   echo "Skip"
else
   export HSA_XNACK=1
   ROCM_GPU=$(rocminfo 2>/dev/null | grep -m 1 -E "gfx[^0]{1}" | sed -e 's/ *Name: *//')
   ROCM_GPU=${ROCM_GPU:-gfx942}

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   SRC_DIR=${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/Reproducers/LCOMPILER-2959
   BUILD_DIR=$(mktemp -d)
   trap 'rm -rf ${BUILD_DIR}' EXIT
   cp ${SRC_DIR}/tp.f90 ${BUILD_DIR}/
   cd ${BUILD_DIR}

   ${FC} -fopenmp --offload-arch=${ROCM_GPU}:xnack+ -O0 tp.f90 -o tp
   ./tp
fi
