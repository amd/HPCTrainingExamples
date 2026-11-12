#!/bin/bash

# LCOMPILER-2446: amdflang mis-parses a named DO construct whose name is `block`
# (block: do ... end do block) when it is the first construct of an OpenMP
# region's structured block, reporting "expected OpenMP construct". Compile-only
# check; no GPU required. PASS means the source parses/compiles.

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
SRC_DIR=${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/Reproducers/LCOMPILER-2446
BUILD_DIR=$(mktemp -d)
trap 'rm -rf ${BUILD_DIR}' EXIT
cp ${SRC_DIR}/named_block_do.f90 ${BUILD_DIR}/
cd ${BUILD_DIR}

if ${FC} -fopenmp --offload-arch=${ROCM_GPU} -c named_block_do.f90 -o named_block_do.o; then
   echo "LCOMPILER2446_PASS"
else
   echo "LCOMPILER2446_FAIL: named DO construct block: mis-parsed inside OpenMP region"
fi
