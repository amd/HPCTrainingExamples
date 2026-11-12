#!/bin/bash

# LCOMPILER-1858: passing arrays as private to a target region errors, but only
# when compiling at -O0 (adding thread_limit(256) hides it for small arrays).
# The failure is at compile time, so this build-only check (compile + device
# link at -O0) does not require a GPU to run. PASS means it builds.

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

ROCM_GPU=$(rocminfo 2>/dev/null | grep -m 1 -E "gfx[^0]{1}" | sed -e 's/ *Name: *//')
ROCM_GPU=${ROCM_GPU:-gfx942}

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRC_DIR=${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/Reproducers/LCOMPILER-1858
BUILD_DIR=$(mktemp -d)
trap 'rm -rf ${BUILD_DIR}' EXIT
cp ${SRC_DIR}/private_arrays.F90 ${BUILD_DIR}/
cd ${BUILD_DIR}

if ${FC} -fopenmp --offload-arch=${ROCM_GPU} -O0 private_arrays.F90 -o private_arrays; then
   echo "LCOMPILER1858_PASS"
else
   echo "LCOMPILER1858_FAIL: private arrays in target region rejected at -O0"
fi
