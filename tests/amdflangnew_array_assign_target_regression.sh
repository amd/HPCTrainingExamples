#!/bin/bash

# DCGPUBU-1552: array-syntax initialization (rhoi(1:nmax) = 0.) inside an OpenMP
# target region fails to link with amdflang-new -- ld.lld: undefined symbol
# _FortranAAssign. The failure is at device link, so this build-only check does
# not require a GPU to run. PASS means it links.

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
SRC_DIR=${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/Reproducers/DCGPUBU-1552
BUILD_DIR=$(mktemp -d)
trap 'rm -rf ${BUILD_DIR}' EXIT
cp ${SRC_DIR}/mwe.f90 ${BUILD_DIR}/
cd ${BUILD_DIR}

if ${FC} -fopenmp --offload-arch=${ROCM_GPU} -O0 -g -cpp mwe.f90 -o x_mwe; then
   echo "DCGPUBU1552_PASS"
else
   echo "DCGPUBU1552_FAIL: undefined symbol _FortranAAssign for array-syntax init in target region"
fi
