#!/bin/bash

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
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

module load hipfort
AMDGPU_GFXMODEL=`rocminfo | grep gfx | sed -e 's/Name://' | head -1 |sed 's/ //g'`

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

# Clone the hipfort sources that MATCH the installed hipfort we link against.
# The build compiles main.f03/hip_implementation.cpp but resolves `use hipfort`
# against the INSTALLED package ($HIPFORT_INC .mod files + -lhipfort-amdgcn from
# $HIPFORT_LIB). Cloning upstream HEAD (the old behavior) pulls a different
# hipfort version, so the build succeeds (the .mod still reads) but the call ABI
# no longer matches the linked library -> a.out segfaults at runtime. hipfort
# tags releases as rocm-X.Y.Z (github.com/ROCm/hipfort/tags), so pin the clone
# to the installed ROCm version. Detect that version from the environment set by
# `module load hipfort` (ROCM_PATH / HIPFORT_LIB / HIPFORT_INC), extracting the
# numeric X.Y.Z (handles rocm-7.2.3 as well as rocm-therock-7.13.0 trees).
HIPFORT_CLONE=hipfort_for_test_rocm_2003
HIPFORT_URL=https://github.com/ROCm/hipfort
ROCM_VER=""
for _cand in "${ROCM_PATH}" "${HIPFORT_LIB%/lib}" "${HIPFORT_INC%/include/hipfort}"; do
   [ -n "${_cand}" ] || continue
   _v=$(echo "${_cand}" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
   if [ -n "${_v}" ]; then ROCM_VER="${_v}"; break; fi
done
if [ -z "${ROCM_VER}" ] && command -v hipcc >/dev/null 2>&1; then
   ROCM_VER=$(readlink -f "$(command -v hipcc)" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
fi
unset _cand _v

if [ -n "${ROCM_VER}" ] && git clone --depth 1 --branch "rocm-${ROCM_VER}" "${HIPFORT_URL}" "${HIPFORT_CLONE}" 2>/dev/null; then
   echo "hipfort: cloned matching release tag rocm-${ROCM_VER}"
else
   echo "hipfort: tag rocm-${ROCM_VER:-<unknown>} unavailable; falling back to default branch (HEAD) -- runtime ABI skew possible"
   rm -rf "${HIPFORT_CLONE}"
   git clone --depth 1 "${HIPFORT_URL}" "${HIPFORT_CLONE}"
fi

pushd hipfort_for_test_rocm_2003/test/f2003/vecadd

HIPFORT_COMP=`which amdflang`

# Try example from source director
hipfc -v --offload-arch=${AMDGPU_GFXMODEL} -hipfort-compiler $HIPFORT_COMP  hip_implementation.cpp main.f03
./a.out
