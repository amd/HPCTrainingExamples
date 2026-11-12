#!/bin/bash

# ROCM-31493: on ROCm 10 the rocprofiler-sdk public headers transitively pull in
# <sys/socket.h>, whose POSIX `int shutdown(int,int)` collides with a consumer's
# own `shutdown` identifier (as in HPCToolkit's rocm-ss.c). Compile-only check;
# no GPU required. PASS means the translation unit compiles clean.

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   [ -z "$CC" ] && export CC=`which cc`
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
fi
CC=${CC:-amdclang}
command -v ${CC} >/dev/null 2>&1 || CC=cc

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRC_DIR=${REPO_DIR}/rocprofiler-sdk/Reproducer/ROCM-31493
BUILD_DIR=$(mktemp -d)
trap 'rm -rf ${BUILD_DIR}' EXIT
cp ${SRC_DIR}/* ${BUILD_DIR}/
cd ${BUILD_DIR}

if ${CC} -I${ROCM_PATH}/include -c rocm_ss_shutdown.c -o rocm_ss_shutdown.o; then
   echo "ROCM31493_PASS"
else
   echo "ROCM31493_FAIL: rocprofiler-sdk headers collide with a local shutdown identifier"
fi
