#!/bin/bash

# ROCM-33903: AMDGPU StructurizeCFG rewrites a phi incoming value to poison, so
# lanes on the divergent "else" path compute a 64-bit store index with stale
# upper bits and store out of bounds -> GPU page fault at -O1+ (ROCm 7.2.0+).
# Workaround: -mllvm -simplifycfg-sink-common=false.

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
   SRC_DIR=${REPO_DIR}/HIP/Reproducers/ROCM-33903
   BUILD_DIR=$(mktemp -d)
   trap 'rm -rf ${BUILD_DIR}' EXIT
   cp ${SRC_DIR}/* ${BUILD_DIR}/
   cd ${BUILD_DIR}

   ${HIPCC} --offload-arch=${ROCM_GPU} -O2 miscompile.cpp -o miscompile
   ./miscompile
fi
