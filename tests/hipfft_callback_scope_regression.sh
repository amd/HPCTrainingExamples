#!/bin/bash

# ROCM-33905: hipfftXtSetCallback stores the address of the caller's callback
# arrays instead of copying their contents. A stack-local callback array that
# goes out of scope before hipfftExecR2C then yields a GPU memory fault (7.2.0+)
# or silently wrong results. Passes on ROCm <= 7.1.1 (cuFFT copy semantics).

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   [ -z "$HIPCC" ] && export HIPCC=`which hipcc`
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
fi
HIPCC=${HIPCC:-hipcc}

XNACK_COUNT=`rocminfo | grep xnack | wc -l`
if [ ${XNACK_COUNT} -lt 1 ]; then
   echo "Skip"
else
   ROCM_GPU=$(rocminfo 2>/dev/null | grep -m 1 -E "gfx[^0]{1}" | sed -e 's/ *Name: *//')
   ROCM_GPU=${ROCM_GPU:-gfx942}

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   SRC_DIR=${REPO_DIR}/Libraries/FFT/Reproducers/ROCM-33905
   BUILD_DIR=$(mktemp -d)
   trap 'rm -rf ${BUILD_DIR}' EXIT
   cp ${SRC_DIR}/* ${BUILD_DIR}/
   cd ${BUILD_DIR}

   ${HIPCC} --offload-arch=${ROCM_GPU} -O2 -fgpu-rdc --hip-link repro_scope.cpp -lhipfft -o repro_scope
   ./repro_scope clobber
fi
